import streamlit as st
from api.api_model import _get_all_universities_majors, _normalize_toefl_ielts_score, _predict_acceptance, _get_all_universities, _get_major_by_university, _get_applied_info
from models.model_university import *
import plotly.express as px

# https://app-demo-jqzfc4d4nb2dm4mkhxqzgr.streamlit.app/

# https://docs.streamlit.io/library/api-reference

"""
# UniFinder

A One-Stop Solution for predicting your admission chances to top universities depending or suggesting universities based on your profile.
"""

dataset = pd.read_csv('data\TOEFL_IELTS_Combined.csv')
tab1, tab2 = st.tabs([
    'Predict Admission', 'Suggest Universities'
])

# This Tab is for predicting admission chances or App 1
with tab1:
    user_input_submitted = False

    if 'selected_university' not in st.session_state:
        st.session_state.selected_university = None

    # 5 categories we want to currently evaluate: 'GPA', 'GRE Total', 'TOEFL', 'Work Exp', 'Papers'
    with st.form("user_input"):
        st.write(":red[\* Indicates required question]", )
        
        UNIVERSITY_NAME = st.selectbox(options=_get_all_universities(), label="Select your University/Major to apply")
        UNIVERSITY_MAJOR = st.selectbox(options=_get_major_by_university(), label="Select your Major to apply")
        SEASON = st.selectbox(options=["Fall", "Spring"], label="Select the season to apply")
        GPA = st.number_input("Enter GPA score* [0.0 - 4.0]:", value=3.0, min_value=1.0, max_value=4.0, step=0.1, placeholder="0.0 - 4.0")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            GRE_VERBAL = st.number_input("GRE Verbal score* [130 - 170]: ", value=150, min_value=130, max_value=170, step=1, placeholder="130 - 170")
        with col2:
            GRE_QUANTITATIVE = st.number_input("GRE Quantitative score* [130 - 170]: ", value=150, min_value=130, max_value=170, step=1, placeholder="130 - 170")
        with col3:
            GRE_WRITING = st.number_input("GRE Writing score* [0.0 - 6.0]: ", value=3.0, min_value=0.0, max_value=6.0, step=0.1, placeholder="0.0 - 6.0")
        with col4:
            GRE_TOTAL = st.number_input("GRE Total score* [260 - 340]: ", value=300, min_value=260, max_value=340, step=1, placeholder="260 - 340")

        TOEFL_IELTS = st.number_input("Enter TOEFL [0-120] / IELTS [0-9] score*: ", value=100, min_value=0, max_value=120, step=1, placeholder="TOEFL [0-120] or IELTS [0-9]")
        PAPERS = st.number_input("Enter number of papers published: ", value=1, min_value=0, max_value=100, step=1)
        WORK_EXPERIENCE = st.number_input("Working experience (months): ",  min_value=0, max_value=100, step=1)

        submit_university_name_input = st.form_submit_button(label="Predict Admission")
        
        if submit_university_name_input:
            NORMALIZED_TOEFL_IELTS = _normalize_toefl_ielts_score(TOEFL_IELTS)
            #st.write(NORMALIZED_TOEFL_IELTS)
            student_data = {
                "University": UNIVERSITY_NAME,
                "Major": UNIVERSITY_MAJOR,
                "GPA": GPA,
                "GRE Verbal": GRE_VERBAL,
                "GRE Quantitative": GRE_QUANTITATIVE,
                "GRE Writing": GRE_WRITING,
                "GRE Total": GRE_TOTAL,
                "TOEFL/IELTS": NORMALIZED_TOEFL_IELTS,
                "Papers": PAPERS,
                "Work Exp": WORK_EXPERIENCE,
                "Season": SEASON
            }
            
            
            admit_status = get_prediction(student_data)
            
            if admit_status == 1:
                st.info(f"You are likely to get admitted at {UNIVERSITY_NAME} for {UNIVERSITY_MAJOR}.")
            else:
                st.warning(f"You are less likely to get admitted at {UNIVERSITY_NAME} for {UNIVERSITY_MAJOR}.")
            
            results = _get_applied_info(UNIVERSITY_NAME, UNIVERSITY_MAJOR)

            attributes = ["GPA", "GRE Verbal", "GRE Quantitative", "GRE Writing", "GRE Total", "TOEFL/IELTS", "Papers", "Work Exp"]

            for attribute in attributes:
                fig = px.histogram(results, x=attribute, title=f"{attribute} Scores for {UNIVERSITY_NAME} - {UNIVERSITY_MAJOR}")
                fig.add_vline(x=student_data[attribute], line_dash="solid", line_color="red", annotation_text=f"Your {attribute}", annotation_position="top right")
                
                st.plotly_chart(fig)
            


# This Tab is for suggesting universities or App 2
with tab2:
    user_input_submitted = False

    if 'selected_university' not in st.session_state:
        st.session_state.selected_university = None

    # 5 categories we want to currently evaluate: 'GPA', 'GRE Total', 'TOEFL', 'Work Exp', 'Papers'
    with st.form("user_input_suggestion"):
        st.write(":red[\* Indicates required question]", )
        
        UNIVERSITY_MAJOR = st.selectbox(options=_get_major_by_university(), label="Select your Major to apply")
        SEASON = st.selectbox(options=["Fall", "Spring"], label="Select the season to apply")
        GPA = st.number_input("Enter GPA score* [0.0 - 4.0]:", value=3.0, min_value=1.0, max_value=4.0, step=0.1, placeholder="0.0 - 4.0")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            GRE_VERBAL = st.number_input("GRE Verbal score* [130 - 170]: ", value=150, min_value=130, max_value=170, step=1, placeholder="130 - 170")
        with col2:
            GRE_QUANTITATIVE = st.number_input("GRE Quantitative score* [130 - 170]: ", value=150, min_value=130, max_value=170, step=1, placeholder="130 - 170")
        with col3:
            GRE_WRITING = st.number_input("GRE Writing score* [0.0 - 6.0]: ", value=3.0, min_value=0.0, max_value=6.0, step=0.1, placeholder="0.0 - 6.0")
        with col4:
            GRE_TOTAL = st.number_input("GRE Total score* [260 - 340]: ", value=300, min_value=260, max_value=340, step=1, placeholder="260 - 340")

        TOEFL_IELTS = st.number_input("Enter TOEFL [0-120] / IELTS [0-9] score*: ", value=100, min_value=0, max_value=120, step=1, placeholder="TOEFL [0-120] or IELTS [0-9]")
        PAPERS = st.number_input("Enter number of papers published: ", value=1, min_value=0, max_value=100, step=1)
        WORK_EXPERIENCE = st.number_input("Working experience (months): ",  min_value=0, max_value=100, step=1)

        submit_university_name_input = st.form_submit_button(label="Suggest Universities")
        
        if submit_university_name_input:
            NORMALIZED_TOEFL_IELTS = _normalize_toefl_ielts_score(TOEFL_IELTS)
            #st.write(NORMALIZED_TOEFL_IELTS)
            student_data = {
                "University": UNIVERSITY_NAME,
                "Major": UNIVERSITY_MAJOR,
                "GPA": GPA,
                "GRE Verbal": GRE_VERBAL,
                "GRE Quantitative": GRE_QUANTITATIVE,
                "GRE Writing": GRE_WRITING,
                "GRE Total": GRE_TOTAL,
                "TOEFL/IELTS": NORMALIZED_TOEFL_IELTS,
                "Papers": PAPERS,
                "Work Exp": WORK_EXPERIENCE,
                "Season": SEASON
            }
            
            
            admit_status = get_prediction(student_data)
            
            if admit_status == 1:
                st.info(f"You are likely to get admitted at {UNIVERSITY_NAME} for {UNIVERSITY_MAJOR}.")
            else:
                st.warning(f"You are less likely to get admitted at {UNIVERSITY_NAME} for {UNIVERSITY_MAJOR}.")
                
            # st.write(student_data)
            # st.write(UNIVERSITY_NAME)
            #st.write(_predict_acceptance(student_data, UNIVERSITY_NAME_MAJOR))   