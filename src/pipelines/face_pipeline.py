

import dlib
import numpy as np
import face_recognition_models
from sklearn.svm import SVC
import streamlit as st
from PIL import Image, ImageOps

from src.database.db import get_all_students


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    sp = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    facerec = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, sp, facerec


def detect_faces_robust(image_np):
    # Use the CACHED detector — not a new one every call
    detector, sp, facerec = load_dlib_models()

    # 1. Try original size (upsample 0) — best for close-up / large faces
    faces = detector(image_np, 0)
    if len(faces) > 0:
        return faces, image_np

    # 2. Try original size (upsample 1) — best for small / distant faces
    faces = detector(image_np, 1)
    if len(faces) > 0:
        return faces, image_np

    pil_img = Image.fromarray(image_np)

    # 3. Try downscaling to standard widths — HOG detector is scale-sensitive
    for scale_width in [800, 600, 400]:
        if scale_width < pil_img.size[0]:
            w_percent = scale_width / float(pil_img.size[0])
            h_size = int(float(pil_img.size[1]) * w_percent)
            resized_pil = pil_img.resize((scale_width, h_size), Image.Resampling.LANCZOS)
            resized_np = np.array(resized_pil)

            faces = detector(resized_np, 0)
            if len(faces) > 0:
                return faces, resized_np

            faces = detector(resized_np, 1)
            if len(faces) > 0:
                return faces, resized_np

            # Equalize contrast and try again
            eq_pil = ImageOps.equalize(resized_pil)
            eq_np = np.array(eq_pil)

            faces = detector(eq_np, 0)
            if len(faces) > 0:
                return faces, eq_np

            faces = detector(eq_np, 1)
            if len(faces) > 0:
                return faces, eq_np

    # 4. Equalize original size and try
    eq_pil_img = ImageOps.equalize(pil_img)
    eq_image_np = np.array(eq_pil_img)

    faces = detector(eq_image_np, 0)
    if len(faces) > 0:
        return faces, eq_image_np

    faces = detector(eq_image_np, 1)
    if len(faces) > 0:
        return faces, eq_image_np

    return [], image_np


def get_face_embeddings(image_np):
    detector, sp, facerec = load_dlib_models()

    # Use robust multi-scale and multi-stage detector
    faces, detected_image_np = detect_faces_robust(image_np)

    encodings = []
    for face in faces:
        shape = sp(detected_image_np, face)
        face_descriptor = facerec.compute_face_descriptor(detected_image_np, shape, 1)  # 128-dim embedding
        encodings.append(np.array(face_descriptor))

    return encodings


@st.cache_resource
def get_trained_model():
    X = []
    y = []

    student_db = get_all_students()

    if not student_db:
        return None

    for student in student_db:
        embedding = student.get('face_embedding')
        if embedding:
            X.append(np.array(embedding))
            y.append(student.get('student_id'))

    if len(X) == 0:
        return None

    clf = SVC(kernel='linear', class_weight='balanced')

    try:
        clf.fit(X, y)
    except ValueError:
        return None

    return {'clf': clf, 'X': X, 'y': y}


def train_classifier():
    st.cache_resource.clear()
    model_data = get_trained_model()
    return bool(model_data)


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)
    num_faces = len(encodings)
    detected_student = {}

    if num_faces == 0:
        return detected_student, [], 0

    model_data = get_trained_model()

    # If no students are registered yet, still return num_faces so the UI
    # knows a face WAS detected (and can show the registration form)
    if not model_data:
        return detected_student, [], num_faces

    clf = model_data['clf']
    X_train = model_data['X']
    y_train = model_data['y']

    all_students = sorted(list(set(y_train)))

    for encoding in encodings:
        if len(all_students) >= 2:
            predicted_id = int(clf.predict([encoding])[0])
        else:
            predicted_id = int(all_students[0])

        student_embedding = X_train[y_train.index(predicted_id)]
        best_match_score = np.linalg.norm(student_embedding - encoding)

        resemblance_threshold = 0.6

        if best_match_score <= resemblance_threshold:
            detected_student[predicted_id] = True

    return detected_student, all_students, num_faces
