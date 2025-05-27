import os
import math
import glob
import numpy as np

def calculate_bbox_width(bbox):
    x1, y1, x2, y2 = bbox
    return x2 - x1

def calculate_distance(bbox1, bbox2):
    center1 = [(bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2]
    center2 = [(bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2]
    return math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

def calibrate_thresholds(calibration_pattern, model, class_names):
    matching_dirs = glob.glob(calibration_pattern)
    if not matching_dirs:
        raise FileNotFoundError(f"No directories found matching pattern: {calibration_pattern}")
    print(f"Found {len(matching_dirs)} matching folders for calibration")

    all_image_files = []
    for directory in matching_dirs:
        images_dir = os.path.join(directory, 'images')
        if not os.path.exists(images_dir):
            images_dir = directory
        if os.path.exists(images_dir):
            for f in os.listdir(images_dir):
                if f.endswith(('.jpg', '.jpeg', '.png')):
                    all_image_files.append(os.path.join(images_dir, f))

    widths = {'monitor': [], 'keyboard': [], 'mouse': []}
    distances = {'keyboard_monitor': [], 'mouse_keyboard': [], 'monitor_mouse': []}

    for image_path in all_image_files:
        results = model(image_path)[0]
        bboxes = {}
        for box in results.boxes:
            class_id = int(box.cls)
            class_name = class_names[class_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bboxes[class_name] = (x1, y1, x2, y2)

        if all(k in bboxes for k in widths):
            for key in widths:
                widths[key].append(calculate_bbox_width(bboxes[key]))

            distances['keyboard_monitor'].append(calculate_distance(bboxes['keyboard'], bboxes['monitor']))
            distances['mouse_keyboard'].append(calculate_distance(bboxes['mouse'], bboxes['keyboard']))
            distances['monitor_mouse'].append(calculate_distance(bboxes['monitor'], bboxes['mouse']))

    reference_widths = {k: np.mean(v) if v else 0 for k, v in widths.items()}
    reference_distances = {k: np.mean(v) if v else 0 for k, v in distances.items()}

    print("Calibration complete.")
    return reference_widths, reference_distances

def deviation_text(measured, reference, label, threshold=0.1):
    if reference == 0:
        return f"{label}: No reference value."
    deviation = abs(measured - reference) / reference
    if deviation > threshold:
        return f"{label}: Deviation of {deviation*100:.1f}% (measured {measured:.1f} vs reference {reference:.1f}) — Not OK"
    else:
        return f"{label}: Within acceptable range ({deviation*100:.1f}% deviation)"

def evaluate_seiton(results, reference_widths, reference_distances, width_threshold=0.2, distance_threshold=0.2):
    seiton_reasons = []
    bboxes = {}
    flipped_objects = []

    for box in results.boxes:
        class_id = int(box.cls)
        class_name = results.names[class_id]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bboxes[class_name] = [x1, y1, x2, y2]
        if 'flipped' in class_name:
            flipped_objects.append(class_name)

    for fo in flipped_objects:
        seiton_reasons.append(f"Flipped {fo}")

    for obj_name in ['monitor', 'keyboard', 'mouse']:
        for variant in [obj_name, f"{obj_name}-flipped"]:
            if variant in bboxes:
                width = calculate_bbox_width(bboxes[variant])
                ref_width = reference_widths.get(obj_name, 0)
                print(f"Checking {variant}: measured width = {width:.2f}, reference = {ref_width:.2f}")
                if ref_width > 0:
                    diff_ratio = (width - ref_width) / ref_width
                    print(f"Width diff ratio for {variant}: {diff_ratio:.3f}")
                    if abs(diff_ratio) > width_threshold:
                        direction = "large" if diff_ratio > 0 else "small"
                        seiton_reasons.append(f"{variant} width too {direction} ({abs(diff_ratio)*100:.1f}% deviation)")
            break


    pairs = [
        ('keyboard', 'monitor', 'keyboard_monitor'),
        ('mouse', 'keyboard', 'mouse_keyboard'),
        ('monitor', 'mouse', 'monitor_mouse')
    ]

    for obj1, obj2, dist_key in pairs:
        obj1_bbox = None
        obj2_bbox = None
        for v1 in [obj1, f"{obj1}-flipped"]:
            if v1 in bboxes:
                obj1_bbox = bboxes[v1]
                break
        for v2 in [obj2, f"{obj2}-flipped"]:
            if v2 in bboxes:
                obj2_bbox = bboxes[v2]
                break

        if obj1_bbox and obj2_bbox:
            distance = calculate_distance(obj1_bbox, obj2_bbox)
            ref_distance = reference_distances.get(dist_key, 0)
            if ref_distance > 0:
                diff_ratio = (distance - ref_distance) / ref_distance
                if abs(diff_ratio) > distance_threshold:
                    direction = "far" if diff_ratio > 0 else "close"
                    seiton_reasons.append(f"{obj1}-{obj2} distance too {direction} ({abs(diff_ratio)*100:.1f}% deviation)")

    is_seiton = len(seiton_reasons) == 0
    return is_seiton, seiton_reasons