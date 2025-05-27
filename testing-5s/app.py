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
from seiton_utils import calibrate_thresholds, calculate_distance, calculate_bbox_width, evaluate_seiton


# Load models
aura_model = YOLO(r'D:\dynamic5s-video\testing-5s\models\aura.pt')          # Shine model
last_model = YOLO(r'D:\dynamic5s-video\testing-5s\models\yolo11n.pt')          # Sorting model
yolo_model_seiton = YOLO(r'D:\dynamic5s-video\testing-5s\models\bestyolo11s.pt')   # Set in Order model
thresholds = joblib.load(r"D:\dynamic5s-video\testing-5s\models\reference_thresholds.pkl") #Thresholds for Set in Order
REFERENCE_WIDTHS = thresholds['widths']
REFERENCE_DISTANCES = thresholds['distances']


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

REQUIRED_SORT_ITEMS = {
    'keyboard': 0.34,
    'mouse': 0.33,
    'monitor': 0.33
}

# Output dictionary to hold current processing results
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

def image_with_boxes(results):
    # Convert model prediction plot to base64 encoded PNG image
    im_array = results[0].plot()
    im = Image.fromarray(im_array)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def process_model(model, uploaded_file, model_type):
    # Save uploaded file temporarily
    temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
    uploaded_file.save(temp_path)

    results = model(temp_path)
    result = results[0]

    detected_labels = []
    if result.boxes and result.names:
        for cls_id in result.boxes.cls:
            class_id = int(cls_id)
            class_name = result.names[class_id]
            detected_labels.append(class_name)

    image_encoded = image_with_boxes(results)

    if model_type == 'shine':
        output['shine_image'] = image_encoded
        bad_labels = ['coffee_cup', 'coffee_no_trash', 'cookie', 'paper', 'plastic', 'tissue', 'vinyl']
        for label in detected_labels:
            if label in bad_labels:
                return "No, not shine"
        return "Yes, it is shine"

    elif model_type == 'sort':
        output['sorting_image'] = image_encoded
        output['sorting_labels'] = detected_labels

        found_items = {key: False for key in REQUIRED_SORT_ITEMS}

        for label in detected_labels:
            if label in REQUIRED_SORT_ITEMS:
                found_items[label] = True
            elif label == "personal-item":
                output['sorting_score'] = 0.0
                return "It is not sort"

        detected_count = sum(found_items.values())
        score = round(detected_count / 3, 2)
        output['sorting_score'] = score

        missing_items = [item for item, found in found_items.items() if not found]
        if missing_items:
            return f"Item(s) {', '.join(missing_items)} not detected"
        else:
            return "It is sort"

def process_set_in_order(image_file):
    temp_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    image_file.save(temp_path)

    results = yolo_model_seiton(temp_path)[0]
    output['seiton_image'] = image_with_boxes([results])

    classification, reasons = evaluate_seiton(results, REFERENCE_WIDTHS, REFERENCE_DISTANCES)

    return classification, reasons

def create_pie_chart():
    labels = ['Shine', 'Sorting', 'Set in Order', 'Standardize', 'Sustain']
    sizes = [20] * 5

    S_COLORS = {
        'shine': '#FFAAAA',
        'sorting': '#FF8282',
        'set_in_order': '#CD5656',
        'standardize': '#AF3E3E',
        'sustain': '#A62C2C'
    }

    colors = ['black'] * 5

    shine_ok = output['shine'] == "Yes, it is shine"
    sorting_score = output['sorting_score']
    sorting_detected = int(sorting_score * 3)
    set_in_order_ok = output['set_in_order'] == "Yes, it is set in order"
    sustain_ok = output['sustain'] == "Yes"

    if shine_ok:
        colors[0] = S_COLORS['shine']
    if sorting_detected == 3:
        colors[1] = S_COLORS['sorting']
    if set_in_order_ok:
        colors[2] = S_COLORS['set_in_order']

    if shine_ok and sorting_detected == 3 and set_in_order_ok:
        colors[3] = S_COLORS['standardize']

    if sustain_ok:
        colors[4] = S_COLORS['sustain']

    five_s_achieved = shine_ok and sorting_detected == 3 and set_in_order_ok and sustain_ok

    fig, ax = plt.subplots()
    wedges, texts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        startangle=90,
        textprops={'color': 'black'},
        wedgeprops=dict(width=0.4)
    )

    if sorting_detected < 3:
        ax.text(0, -1.35, f"Sorting: {sorting_detected}/3", ha='center', color='black', fontsize=12)

    if five_s_achieved:
        ax.text(0, -1.55, "5S Achieved!", ha='center', color='black', fontsize=13, weight='bold')
    elif shine_ok and sorting_detected == 3 and set_in_order_ok:
        ax.text(0, -1.55, "3S Achieved! Standardize Achieved!", ha='center', color='black', fontsize=13, weight='bold')

    ax.axis('equal')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return img_base64

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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        fullname = request.form.get('fullname')
        employee_id = request.form.get('employee_id')
        phone_number = request.form.get('phone_number')

        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        department = request.form.get('department')
        role = request.form.get('role')
        gender = request.form.get('gender')

        # Password match check
        if password != confirm_password:
            flash("Passwords do not match!", "error")
            return redirect(url_for('register'))

        # You can now save this data to a database or print for testing
        print(f"New registration: {fullname}, {employee_id}, {phone_number}, {email}, {department}, {role}, {gender}")

        flash("Registration successful!", "success")
        return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # You can implement authentication logic here
        email = request.form.get('email')
        password = request.form.get('password')
        # For now, just flash and redirect
        flash(f"Login attempt for user: {email}")
        return redirect(url_for('login'))  # Redirect to login again or change to a dashboard route

    return render_template('login.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')

        # Simulated response, replace with actual email logic later
        print(f"Password reset link requested for: {email}")

        flash("If this email exists, a reset link has been sent.", "info")
        return redirect(url_for('forgot_password'))

    return render_template('forgot-password.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/indicator', methods=['GET', 'POST'])
def indicator():
    pie_chart = None
    seiton_classification = None

    if request.method == 'POST':
        shine_image = request.files['shine_image']
        sorting_image = request.files['sorting_image']
        seiton_image = request.files['seiton_image']

        # Get Sustain checkbox value from form
        sustain_check = request.form.get('sustain_check', 'No')
        output['sustain'] = "Yes" if sustain_check == 'Yes' else "No"

        # Process each model and store results in output dict
        output['shine'] = process_model(aura_model, shine_image, 'shine')
        output['sorting'] = process_model(last_model, sorting_image, 'sort')
        output['set_in_order'], seiton_reasons = process_set_in_order(seiton_image)

        # Seiton classification text
        seiton_classification = "Seiton Classification: Yes" if output['set_in_order'] == "Yes, it is set in order" else "Seiton Classification: No"

        pie_chart = create_pie_chart()

        return render_template(
            'indicator.html',
            pie_chart=pie_chart,
            shine_result=output['shine'],
            sorting_result=output['sorting'],
            set_in_order_result=output['set_in_order'],
            seiton_result=output['set_in_order'],
            sustain_check=output['sustain'],
            seiton_classification=seiton_classification,
            seiton_reasons=seiton_reasons,
            sorting_labels=output['sorting_labels'],
            shine_image=output['shine_image'],
            sorting_image=output['sorting_image'],
            seiton_image=output['seiton_image'],
            sorting_score=output['sorting_score']
        )

    # GET request: initial empty page
    return render_template(
        'indicator.html',
        pie_chart=None,
        shine_result=None,
        sorting_result=None,
        set_in_order_result=None,
        seiton_result=None,
        sustain_check="No",
        seiton_classification=None,
        sorting_labels=[],
        shine_image=None,
        sorting_image=None,
        seiton_image=None,
        sorting_score=0.0
    )
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
        ext = filename.rsplit('.',1)[1].lower()

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
        output['set_in_order_reasons'] = []

        cap = cv2.VideoCapture(video_save_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        annotated_video_path = os.path.join(app.config['VIDEO_FOLDER'], 'annotated_' + filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(annotated_video_path, fourcc, fps, (width, height))

        frame_count = 0
        personal_item_detected = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            results_shine = aura_model(frame)[0]
            results_sorting = last_model(frame)[0]
            results_seiton = yolo_model_seiton(frame)[0]

            sorting_labels = [results_sorting.names[int(cls)] for cls in results_sorting.boxes.cls] if results_sorting.boxes else []
            
            if 'personal-item' in sorting_labels:
                personal_item_detected = True
            
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
                
                required_items = {'mouse','keyboard', 'monitor'}
                missing_items = required_items - set(sorting_labels)
                
                is_seiton, seiton_reasons = evaluate_seiton(results_seiton, REFERENCE_WIDTHS, REFERENCE_DISTANCES)
                if missing_items:
                    is_seiton = False
                    if seiton_reasons is None:
                        seiton_reasons = []
                    for item in missing_items:
                        seiton_reasons.append(f"Missing required item: {item}")

                output['set_in_order_reasons'] = seiton_reasons

                if is_seiton:
                    output['set_in_order'] = "Yes, it is set in order"
                else:
                    if seiton_reasons:
                        reason_text = "; ".join(seiton_reasons)
                        output['set_in_order'] = f"No, not set in order: {reason_text}"
                    else:
                        output['set_in_order'] = "No, not set in order: reasons not specified"


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
            set_in_order_reasons=output['set_in_order_reasons'],
            pie_chart=pie_base64,
            sorting_labels=output['sorting_labels'],
            annotated_frame=output.get('annotated_frame')
        )

    return render_template('video.html')

if __name__ == '__main__':
    app.run(debug=True)
