import os
import io
import base64
import shutil
import numpy as np
import cv2
import torch
import joblib
from PIL import Image
from flask import Flask, render_template, request, redirect, flash, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------ Configuration ------------------
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

UPLOAD_FOLDER = 'uploads'
VIDEO_FOLDER = 'static/videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER

# Allowed extensions for upload
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# ------------------ Load Models ------------------
aura_model = YOLO('aura.pt')             # Shine model
last_model = YOLO('viriya.pt')           # Sorting model
yolo_model_seiton = YOLO('irene.pt')     # Set in Order YOLOv10n model
rf_model_seiton = joblib.load("rf_model_seiton.pkl")  # Random Forest classifier for Seiton

# ------------------ Global State ------------------
REQUIRED_SORT_ITEMS = {
    'keyboard': 0.34,
    'mouse': 0.33,
    'monitor': 0.33
}

output = {
    'shine': None,
    'sorting': None,
    'set_in_order': None,
    'sustain': None,
    'sorting_labels': [],
    'shine_image': None,
    'sorting_image': None,
    'seiton_image': None,
    'sorting_score': 0.0
}

# ------------------ Helper Functions ------------------

def allowed_file(filename, allowed_set):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_set

def image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def results_to_base64_image(results) -> str:
    """Convert YOLO result plot image to base64 string."""
    im_array = results[0].plot()
    im = Image.fromarray(im_array)
    return image_to_base64(im)

def iou(box1, box2):
    """Compute Intersection over Union of two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def merge_and_filter_boxes(all_results):
    """
    Merge boxes from multiple YOLO results, filter duplicates by IoU threshold.

    all_results: list of tuples (result, model_name)
    Returns list of dicts: {'box', 'label', 'confidence', 'model'}
    """
    iou_threshold = 0.5
    merged = []

    for result, model_name in all_results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else np.ones(len(boxes))
        class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else np.zeros(len(boxes), dtype=int)
        names = result.names

        for box, score, cls_id in zip(boxes, scores, class_ids):
            label = names[cls_id]
            merged.append({
                'box': box,
                'label': label,
                'confidence': score,
                'model': model_name
            })

    merged.sort(key=lambda x: x['confidence'], reverse=True)

    filtered = []
    while merged:
        current = merged.pop(0)
        filtered.append(current)
        merged = [
            other for other in merged
            if not (other['label'] == current['label'] and iou(current['box'], other['box']) > iou_threshold)
        ]

    return filtered

def get_object_centers(results, model):
    centers = {}
    for box in results.boxes:
        cls_id = int(box.cls)
        class_name = model.names[cls_id]
        if class_name == 'tv':
            class_name = 'monitor'
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        centers[class_name.lower()] = (cx, cy)
    return centers

def calc_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calc_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return np.arctan2(dy, dx)

def process_shine(upload_path):
    results = aura_model(upload_path)
    image_encoded = results_to_base64_image(results)
    output['shine_image'] = image_encoded

    detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls] if results[0].boxes else []
    bad_labels = {'coffee_cup', 'coffee_no_trash', 'cookie', 'paper', 'plastic', 'tissue', 'vinyl'}

    for label in detected_labels:
        if label in bad_labels:
            output['shine'] = "No, not shine"
            return output['shine']

    output['shine'] = "Yes, it is shine"
    return output['shine']

def process_sorting(upload_path):
    results = last_model(upload_path)
    image_encoded = results_to_base64_image(results)
    output['sorting_image'] = image_encoded

    detected_labels = [results[0].names[int(cls)] for cls in results[0].boxes.cls] if results[0].boxes else []
    output['sorting_labels'] = detected_labels

    found_items = {key: False for key in REQUIRED_SORT_ITEMS}

    for label in detected_labels:
        if label == 'personal-item':
            output['sorting_score'] = 0.0
            output['sorting'] = "It is not sort"
            return output['sorting']
        if label in REQUIRED_SORT_ITEMS:
            found_items[label] = True

    detected_count = sum(found_items.values())
    score = round(detected_count / 3, 2)
    output['sorting_score'] = score

    missing_items = [item for item, found in found_items.items() if not found]
    if missing_items:
        output['sorting'] = f"Item(s) {', '.join(missing_items)} not detected"
    else:
        output['sorting'] = "It is sort"

    return output['sorting']

def process_set_in_order(upload_path):
    yolo_results = yolo_model_seiton(upload_path)[0]
    seiton_image = results_to_base64_image([yolo_results])
    output['seiton_image'] = seiton_image

    centers = get_object_centers(yolo_results, yolo_model_seiton)
    required_objects = ['mouse', 'monitor', 'keyboard']

    missing_objects = [obj for obj in required_objects if obj not in centers]
    if missing_objects:
        output['set_in_order'] = f"Missing object(s): {', '.join(missing_objects)}"
        return output['set_in_order']

    features = [
        calc_distance(centers['mouse'], centers['monitor']),
        calc_distance(centers['mouse'], centers['keyboard']),
        calc_distance(centers['monitor'], centers['keyboard']),
        calc_angle(centers['mouse'], centers['monitor']),
        calc_angle(centers['mouse'], centers['keyboard']),
        calc_angle(centers['monitor'], centers['keyboard']),
    ]

    prediction = rf_model_seiton.predict([features])
    output['set_in_order'] = "Yes, it is set in order" if prediction[0] == 1 else "No, not set in order"
    return output['set_in_order']

def create_pie_chart():
    labels = ['Shine', 'Sorting', 'Set in Order', 'Sustain']
    # Convert output keys to pie values (assign 1 for 'Yes', 0 for 'No' or missing)
    vals = []
    vals.append(1 if output['shine'] == "Yes, it is shine" else 0)
    vals.append(output.get('sorting_score', 0))
    vals.append(1 if output['set_in_order'] == "Yes, it is set in order" else 0)
    vals.append(0)  # Sustain is not evaluated here, you can extend later

    # Remove zero values & labels to clean up pie chart
    filtered_labels_vals = [(label, val) for label, val in zip(labels, vals) if val > 0]
    if not filtered_labels_vals:
        filtered_labels_vals = [('No Detection', 1)]

    labels, vals = zip(*filtered_labels_vals)

    fig, ax = plt.subplots(figsize=(5,5))
    wedges, texts, autotexts = ax.pie(vals, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff','#99ff99','#ffcc99'])
    plt.axis('equal')

    # Save pie chart to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def draw_boxes_on_frame(frame, merged_boxes):
    """Draw bounding boxes and labels on a frame with colors (color can be fixed or random)."""
    # You can keep colors or unify them if you want:
    COLOR = (0, 255, 0)  # Green for all boxes (or change to any color you like)

    for obj in merged_boxes:
        x1, y1, x2, y2 = map(int, obj['box'])
        label = obj['label']
        confidence = obj['confidence']

        # Format label with confidence, e.g. "keyboard 0.87"
        label_with_conf = f"{label} {confidence:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR, 2)
        (text_w, text_h), _ = cv2.getTextSize(label_with_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_h - 6), (x1 + text_w, y1), COLOR, -1)
        cv2.putText(frame, label_with_conf, (x1, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame


# ------------------ Routes ------------------

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Clear previous outputs
        for key in output:
            output[key] = None
        output['sorting_labels'] = []
        output['sorting_score'] = 0.0

        # Check uploaded file
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()

        if allowed_file(filename, ALLOWED_IMAGE_EXTENSIONS):
            # Save image
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Process each 5S step
            process_shine(save_path)
            process_sorting(save_path)
            process_set_in_order(save_path)

            pie_base64 = create_pie_chart()

            # Remove saved file to keep folder clean
            os.remove(save_path)

            return render_template('index.html', 
                                   shine=output['shine'], 
                                   sorting=output['sorting'], 
                                   set_in_order=output['set_in_order'],
                                   pie_chart=pie_base64,
                                   shine_img=output['shine_image'],
                                   sorting_img=output['sorting_image'],
                                   seiton_img=output['seiton_image'],
                                   sorting_labels=output['sorting_labels'])
        
        elif allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
            # Video upload: save and redirect to video processing
            video_save_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
            file.save(video_save_path)
            flash('Video uploaded, processing...')
            return redirect(f'/video_process/{filename}')

        else:
            flash('Unsupported file type')
            return redirect(request.url)

    # GET request - just render the form
    return render_template('index.html')

@app.route('/video', methods=['GET', 'POST'])
def video():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        ext = filename.rsplit('.', 1)[1].lower()

        if not allowed_file(filename, ALLOWED_VIDEO_EXTENSIONS):
            flash('Unsupported video format')
            return redirect(request.url)

        video_save_path = os.path.join(app.config['VIDEO_FOLDER'], filename)
        file.save(video_save_path)

        output['shine'] = None
        output['sorting'] = None
        output['set_in_order'] = None
        output['sorting_score'] = 0.0
        output['sorting_labels'] = []

        cap = cv2.VideoCapture(video_save_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        annotated_video_path = os.path.join(app.config['VIDEO_FOLDER'], f"annotated_{filename}")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        frame_count = 0
        personal_item_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run YOLO models
            results_shine = aura_model(frame)[0]
            results_sorting = last_model(frame)[0]
            results_seiton = yolo_model_seiton(frame)[0]

            sorting_labels = [results_sorting.names[int(cls)] for cls in results_sorting.boxes.cls] if results_sorting.boxes else []
            if 'personal-item' in sorting_labels:
                personal_item_detected = True

            # Analyze only the first frame for scoring
            if frame_count == 1:
                output['sorting_labels'] = sorting_labels

                if personal_item_detected:
                    output['sorting'] = "It is not sort (personal item detected)"
                    output['sorting_score'] = 0.0
                    output['shine'] = "No, not shine (personal item detected)"
                else:
                    bad_labels = {'coffee_cup', 'coffee_no_trash', 'cookie', 'paper', 'plastic', 'tissue', 'vinyl'}
                    shine_labels = [results_shine.names[int(cls)] for cls in results_shine.boxes.cls] if results_shine.boxes else []
                    output['shine'] = "No, not shine" if any(label in bad_labels for label in shine_labels) else "Yes, it is shine"

                    required = REQUIRED_SORT_ITEMS.keys()
                    found_items = {key: False for key in required}
                    for label in sorting_labels:
                        if label in found_items:
                            found_items[label] = True

                    detected_count = sum(found_items.values())
                    score = round(detected_count / len(required), 2)
                    output['sorting_score'] = score
                    missing = [item for item, found in found_items.items() if not found]
                    output['sorting'] = f"Item(s) {', '.join(missing)} not detected" if missing else "It is sort"

                centers = get_object_centers(results_seiton, yolo_model_seiton)
                required_objects = ['mouse', 'monitor', 'keyboard']
                missing_objects = [obj for obj in required_objects if obj not in centers]
                if missing_objects:
                    output['set_in_order'] = f"Missing object(s): {', '.join(missing_objects)}"
                else:
                    features = [
                        calc_distance(centers['mouse'], centers['monitor']),
                        calc_distance(centers['mouse'], centers['keyboard']),
                        calc_distance(centers['monitor'], centers['keyboard']),
                        calc_angle(centers['mouse'], centers['monitor']),
                        calc_angle(centers['mouse'], centers['keyboard']),
                        calc_angle(centers['monitor'], centers['keyboard']),
                    ]
                    prediction = rf_model_seiton.predict([features])
                    output['set_in_order'] = "Yes, it is set in order" if prediction[0] == 1 else "No, not set in order"

                # Save first annotated frame as preview
                preview_boxes = merge_and_filter_boxes([
                    (results_shine, 'shine'),
                    (results_sorting, 'sorting'),
                    (results_seiton, 'set_in_order'),
                ])
                preview_frame = draw_boxes_on_frame(frame.copy(), preview_boxes)
                _, buffer = cv2.imencode('.jpg', preview_frame)
                output['annotated_frame'] = base64.b64encode(buffer).decode('utf-8')

            # Draw boxes for all frames and write to output video
            all_boxes = merge_and_filter_boxes([
                (results_shine, 'shine'),
                (results_sorting, 'sorting'),
                (results_seiton, 'set_in_order'),
            ])
            annotated_frame = draw_boxes_on_frame(frame.copy(), all_boxes)
            out.write(annotated_frame)

        cap.release()
        out.release()

        pie_base64 = create_pie_chart()

        return render_template(
            'video.html',
            video_file=f"annotated_{filename}",
            shine=output['shine'],
            sorting=output['sorting'],
            set_in_order=output['set_in_order'],
            pie_chart=pie_base64,
            sorting_labels=output['sorting_labels'],
            annotated_frame=output.get('annotated_frame')
        )

    return render_template('video.html')

# ------------------ Run ------------------
if __name__ == '__main__':
    app.run(debug=True)
