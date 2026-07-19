import streamlit as st

from src.ui.base_layout import style_background_dashboard, style_base_layout

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import get_all_students, create_student, get_student_subjects, get_student_attendance, unenroll_student_to_subject
import time

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"""Welcome, {student_data['name']} """)
        if st.button("Logout", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data
            # Clear registration state too
            st.session_state.pop('show_registration', None)
            st.session_state.pop('captured_photo', None)
            st.rerun()

    st.space()

    c1, c2 = st.columns(2)
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        if st.button('Enroll in Subject', type='primary', width='stretch'):
            enroll_dialog()

    st.divider()

    with st.spinner('Loading your enrolled subjects..'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}

    for log in logs:
        sid = log['subject_id']

        if sid not in stats_map:
            stats_map[sid] = {"total": 0, "attended": 0}

        stats_map[sid]['total'] += 1

        if log.get('is_present'):
            stats_map[sid]['attended'] += 1

    cols = st.columns(2)
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']

        stats = stats_map.get(sid, {"total": 0, "attended": 0})

        def unenroll_button():
            if st.button("Unenroll from this course", type='tertiary', width='stretch', icon=':material/delete_forever:'):
                unenroll_student_to_subject(student_id, sid)
                st.toast(f"Unenrolled from {sub['name']} successfully!")
                st.rerun()

        with cols[i % 2]:
            subject_card(
                name=sub['name'],
                code=sub['subject_code'],
                section=sub['section'],
                stats=[
                    ('📅', 'Total', stats['total']),
                    ('✅', 'Attended', stats['attended']),
                ],
                footer_callback=unenroll_button
            )
    footer_dashboard()


def student_screen():

    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return

    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type='secondary', key='loginbackbtn', shortcut="control+backspace"):
            st.session_state['login_type'] = None
            # Clear any registration state when going back
            st.session_state.pop('show_registration', None)
            st.session_state.pop('captured_photo', None)
            st.rerun()

    st.header('Login using FaceID', text_alignment='center')
    st.space()
    st.space()

    # Use session_state to persist registration state across re-runs
    if 'show_registration' not in st.session_state:
        st.session_state.show_registration = False

    photo_source = st.camera_input("Position your face in the center")

    # When a new photo is taken, run the face scan
    if photo_source:
        # Reset registration state when a new photo is taken
        st.session_state.show_registration = False
        st.session_state.pop('captured_photo', None)

        img = np.array(Image.open(photo_source).convert('RGB'))

        with st.spinner('🔍 AI is scanning your face...'):
            try:
                detected, all_ids, num_faces = predict_attendance(img)
            except Exception as e:
                st.error(f'An error occurred during scanning: {e}')
                st.stop()

        if num_faces == 0:
            st.warning('⚠️ No face detected! Please make sure your face is clearly visible and well-lit, then take another photo.')

        elif num_faces > 1:
            st.warning('⚠️ Multiple faces detected! Please ensure only your face is in the frame.')

        else:
            # Exactly 1 face found
            if detected:
                # Known face — log them in
                student_id = list(detected.keys())[0]
                all_students = get_all_students()
                student = next((s for s in all_students if s['student_id'] == student_id), None)

                if student:
                    st.session_state.is_logged_in = True
                    st.session_state.user_role = 'student'
                    st.session_state.student_data = student
                    st.toast(f"Welcome Back {student['name']} 👋")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error('Student record not found. Please re-register.')
            else:
                # Face found but not in the database — show registration
                st.info('👤 Face not recognized! You might be a new student. Please register below.')
                st.session_state.show_registration = True
                # Save the photo bytes so we can use it in registration
                photo_source.seek(0)
                st.session_state.captured_photo = photo_source.read()

    # Show registration form if needed (persists across re-runs via session_state)
    if st.session_state.get('show_registration') and st.session_state.get('captured_photo'):
        with st.container(border=True):
            st.header('Register New Profile')
            st.info('Your face has been captured. Fill in your details below to create your account.')

            new_name = st.text_input("Enter your name", placeholder='E.g. Lohit Kumar')

            st.subheader('Optional: Voice Enrollment')
            st.caption("Record your voice to enable voice-only attendance in the future.")

            audio_data = None
            try:
                audio_data = st.audio_input('Say a short phrase like "My name is Lohit"')
            except Exception:
                st.warning('Audio input not available on this device.')

            if st.button('✅ Create Account', type='primary', width='stretch'):
                if not new_name.strip():
                    st.warning('Please enter your name!')
                else:
                    with st.spinner('Creating your profile...'):
                        try:
                            # Re-load the saved photo from session_state
                            from io import BytesIO
                            img_bytes = st.session_state.captured_photo
                            img = np.array(Image.open(BytesIO(img_bytes)).convert('RGB'))
                            encodings = get_face_embeddings(img)

                            if encodings:
                                face_emb = encodings[0].tolist()

                                voice_emb = None
                                if audio_data:
                                    try:
                                        voice_emb = get_voice_embedding(audio_data.read())
                                    except Exception:
                                        pass

                                response_data = create_student(new_name.strip(), face_embedding=face_emb, voice_embedding=voice_emb)

                                if response_data:
                                    train_classifier()
                                    # Clear registration state
                                    st.session_state.show_registration = False
                                    st.session_state.pop('captured_photo', None)
                                    # Log the student in immediately
                                    st.session_state.is_logged_in = True
                                    st.session_state.user_role = 'student'
                                    st.session_state.student_data = response_data[0]
                                    st.toast(f'🎉 Profile Created! Welcome, {new_name.strip()}!')
                                    time.sleep(1)
                                    st.rerun()
                                else:
                                    st.error('Failed to save your profile to the database. Please try again.')
                            else:
                                st.error('❌ Could not detect your face in the captured photo. Please go back and take a clearer photo.')
                        except Exception as e:
                            st.error(f'Registration failed: {e}')

    footer_dashboard()