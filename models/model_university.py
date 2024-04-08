import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import joblib
from data import University_Majors_Dict as umd
import gdown
import os

FILENAME = "TOEFL_IELTS_Combined"
# streamlit file path runs it from root directory.
# "../data" will cause error.
FILEPATH = './data/' + FILENAME + '.csv'
df_admitsFYI = pd.read_csv(FILEPATH)

model_url = "https://drive.google.com/uc?id=12b4TZnn3O3vApMQvJTNXFLykBcETWziW"
#model_url = "https://drive.google.com/uc?id=1ktMLwFoAXnvsf-sfEXhGbjwkFQ5c2_aX"
# Check if model is downloaded stack_model.joblib
if not os.path.exists('models/stack_model.joblib'):
    gdown.download(model_url, 'models/stack_model.joblib')
model = joblib.load('models/stack_model.joblib')

# def get_prediction():
#     # University	Target Major	GRE Verbal	GRE Quantitative	GRE Writing	GRE Total	GPA	Papers	Work Exp	Season	TOEFL/IELTS
#     # 36859	20	55	160.0	134.0	2.5	294.0	2.6756	0.0	0	0	6.0
#     data_point = [[20, 55, 160.0, 134.0, 2.5, 294.0, 2.6756, 0.0, 0, 0, 6.0]]
#     return model.predict(data_point)[0]

def get_prediction(student_data):
    university_code = umd.university_dict[student_data['University']]
    major_code = umd.target_major_dict[student_data['Major']]
    season_code = 0 if student_data['Season'] == 'Fall' else 1
    
    data_point = [[
        university_code, 
        major_code, 
        student_data['GRE Verbal'], 
        student_data['GRE Quantitative'], 
        student_data['GRE Writing'], 
        student_data['GRE Total'], 
        student_data['GPA'], 
        student_data['Papers'], 
        student_data['Work Exp'], 
        season_code, 
        student_data['TOEFL/IELTS']
        ]]
    
    print(data_point)
    
    return model.predict(data_point)[0]

def get_uni_major(): 
    return df_admitsFYI['Target Major'].unique()

def get_universities():
    return df_admitsFYI['University'].unique()

def get_top_five_universities():
    university_counts = df_admitsFYI['University'].value_counts()
    top_universities = university_counts.head(5).index.tolist()
    return top_universities

def get_top_majors_by_university(university_name):
    
    top_universities = get_top_five_universities()
    filtered_data = df_admitsFYI[df_admitsFYI['University'].isin(top_universities)]

    # store everything again in a dictionary
    top_majors_per_university = {}
    for university in top_universities:
        # Filter the dataset for the current university
        uni_data = filtered_data[filtered_data['University'] == university]
        # Count the instances of each major
        major_counts = uni_data['Target Major'].value_counts().head(5)
        # Store the results
        top_majors_per_university[university] = major_counts
        

    majors = top_majors_per_university[university_name].keys()
    # for major, _ in top_majors_per_university[university_name].items():
    #     majors.append(major)
    return majors


def get_list_majors_per_universities():
    universities = get_top_five_universities()
    universities_majors_list = []
    for university_name in universities:
        majors = get_top_majors_by_university(university_name)
        for major in majors:
            model_key = f"{university_name} - {major}"
            universities_majors_list.append(model_key)
    
    return universities_majors_list

def get_applied_info(university_name, major_name):
    uni_major = df_admitsFYI[(df_admitsFYI['University'] == university_name) & (df_admitsFYI['Target Major'] == major_name)]
    
    # result = {
    #     "Application Count": uni_major.shape[0],
    #     "Number of Admits": uni_major[uni_major['Status'] == 1].shape[0],
    #     "Average GPA": uni_major['GPA'].mean(),
    #     "Average GRE Verbal": uni_major['GRE Verbal'].mean(),
    #     "Average GRE Quantitative": uni_major['GRE Quantitative'].mean(),
    #     "Average GRE Writing": uni_major['GRE Writing'].mean(),
    #     "Average GRE Total": uni_major['GRE Total'].mean(),
    #     "Average TOEFL/IELTS": uni_major['TOEFL/IELTS'].mean(),
    #     "Average Papers": uni_major['Papers'].mean(),
    #     "Average Work Exp": uni_major['Work Exp'].mean()
    # }
    
    result = {
        "GPA": uni_major['GPA'],
        "GRE Verbal": uni_major['GRE Verbal'],
        "GRE Quantitative": uni_major['GRE Quantitative'],
        "GRE Writing": uni_major['GRE Writing'],
        "GRE Total": uni_major['GRE Total'],
        "TOEFL/IELTS": uni_major['TOEFL/IELTS'],
        "Papers": uni_major['Papers'],
        "Work Exp": uni_major['Work Exp'],
        "Season": uni_major['Season']
    }
    
    return result