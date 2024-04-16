import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import numpy as np
import plotly.graph_objects as go
from data import University_Majors_Dict as umd
import gdown
import os

FILENAME = "TOEFL_IELTS_Combined"
# streamlit file path runs it from root directory.
# "../data" will cause error.
FILEPATH = './data/' + FILENAME + '.csv'
df_admitsFYI = pd.read_csv(FILEPATH)

stack_model_url = "https://drive.google.com/uc?id=12b4TZnn3O3vApMQvJTNXFLykBcETWziW"
decision_tree_model_url = "https://drive.google.com/uc?id=14kMVbN-L2xKSlgNRXu3cJrF6jbSoNb-U"

# Check if model is downloaded stack_model.joblib
if not os.path.exists('models/stack_model.joblib'):
    gdown.download(stack_model_url, 'models/stack_model.joblib')
model = joblib.load('models/stack_model.joblib')

# Check if model is downloaded major_decision_tree_model.joblib
if not os.path.exists('models/major_decision_tree_model.joblib'):
    gdown.download(decision_tree_model_url, 'models/major_decision_tree_model.joblib')
suggest_uni_model = joblib.load('models/major_decision_tree_model.joblib')

"""
purpose: normalize the test scores from 0 - 9.
return: normalized score from 0 to 9 (integer)
"""
def normalize_toefl_ielts_score(score: int) -> int:
    normalized_score = 0
    
    if score <= 9:  #IELTS
        normalized_score = round((score / 9) * 9, 2)
    else:  #TOEFL
        normalized_score = round((score / 120) * 9, 2)
        
    return normalized_score

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
    # return df_admitsFYI['Target Major'].unique()
    return umd.target_major_dict.keys()   

def get_universities():
    # return df_admitsFYI['University'].unique()
    return umd.university_dict.keys()


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

def get_admit_info(university_name, major_name):
    uni_major = df_admitsFYI[(df_admitsFYI['University'] == university_name) & (df_admitsFYI['Target Major'] == major_name) & (df_admitsFYI['Status'] == 1)]
    

    result = {
        "GPA": uni_major['GPA'].mean(),
        "GRE Verbal": uni_major['GRE Verbal'].mean(),
        "GRE Quantitative": uni_major['GRE Quantitative'].mean(),
        "GRE Writing": uni_major['GRE Writing'].mean(),
        "GRE Total": uni_major['GRE Total'].mean(),
        "TOEFL/IELTS": uni_major['TOEFL/IELTS'].mean(),
        "Papers": uni_major['Papers'].mean(),
        "Work Exp": uni_major['Work Exp'].mean()
        # "Season": uni_major['Season'].mean()
    }    
    return result


def get_school_recommendations(student_data):

    major_code = umd.target_major_dict[student_data['Major']]
    season_code = 0 if student_data['Season'] == 'Fall' else 1

    feat_imp_df = df_admitsFYI.copy()
    feat_imp_df = feat_imp_df.drop(columns=['UG College', 'UG Major','Year'])
    feat_imp_df['Season'] = feat_imp_df['Season'].map({'Fall': 0, 'Spring': 1})
    major_encoder = LabelEncoder()
    major_encoder.fit(feat_imp_df['Target Major'])
    feat_imp_df['Target Major'] = major_encoder.transform(feat_imp_df['Target Major'])
    X = feat_imp_df.iloc[:, 2:]

    # extracting the importances
    feat_imp = pd.DataFrame(suggest_uni_model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

    # start first by multiplying each feature by its corresponding weight
    feat_imp_df['Weighted_Score'] = (
        
        feat_imp_df['Target Major'] * feat_imp.loc['Target Major', 'Importance'] +
        feat_imp_df['GPA'] * feat_imp.loc['GPA', 'Importance'] +
        feat_imp_df['GRE Verbal'] * feat_imp.loc['GRE Verbal', 'Importance'] +
        feat_imp_df['GRE Quantitative'] * feat_imp.loc['GRE Quantitative', 'Importance'] +
        feat_imp_df['GRE Writing'] * feat_imp.loc['GRE Writing', 'Importance'] +
        feat_imp_df['GRE Total'] * feat_imp.loc['GRE Total', 'Importance'] +
        feat_imp_df['TOEFL/IELTS'] * feat_imp.loc['TOEFL/IELTS', 'Importance'] +
        feat_imp_df['Work Exp'] * feat_imp.loc['Work Exp', 'Importance'] +
        feat_imp_df['Season'] * feat_imp.loc['Season', 'Importance']+
        feat_imp_df['Papers'] * feat_imp.loc['Papers', 'Importance']
    )

    # Calculating the average weighted score for students in each university
    average_scores_by_university = feat_imp_df.groupby('University')['Weighted_Score'].mean().sort_values(ascending=False)


    data_point = [[ 
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

    keys = ['major_code', 'GRE Verbal', 'GRE Quantitative', 'GRE Writing', 'GRE Total', 'GPA', 'Papers', 'Work Exp', 'Season', 'TOEFL/IELTS']

    # Convert the list of lists into a dictionary
    student_details = dict(zip(keys, data_point[0]))


    # Calculating the student's weighted score using the feature importances

    student_weighted_score = (
        student_details['major_code'] * feat_imp.loc['Target Major', 'Importance'] +
        student_details['GPA'] * feat_imp.loc['GPA', 'Importance'] +
        student_details['GRE Verbal'] * feat_imp.loc['GRE Verbal', 'Importance'] +
        student_details['GRE Quantitative'] * feat_imp.loc['GRE Quantitative', 'Importance'] +
        student_details['GRE Writing'] * feat_imp.loc['GRE Writing', 'Importance'] +
        student_details['GRE Total'] * feat_imp.loc['GRE Total', 'Importance'] +
        student_details['TOEFL/IELTS'] * feat_imp.loc['TOEFL/IELTS', 'Importance'] +
        student_details['Work Exp'] * feat_imp.loc['Work Exp', 'Importance'] +
        student_details['Season'] * feat_imp.loc['Season', 'Importance']+
        student_details['Papers'] * feat_imp.loc['Papers', 'Importance']
    )

    # Finding the 5 schools above and 5 below the student's score
    schools_above = average_scores_by_university[average_scores_by_university > student_weighted_score].nsmallest(5).sort_values(ascending=False)
    schools_below = average_scores_by_university[average_scores_by_university < student_weighted_score].nlargest(5)

    total_unis = []
    for i in range(len(schools_above.index.tolist())):
        total_unis.append(schools_above.index.tolist()[i])
    # print()
    for i in range(len(schools_below.index.tolist())):
        total_unis.append(schools_below.index.tolist()[i])

    # print(total_unis)
    return(total_unis)

def get_school_recommendations_figure():

    feat_imp_df = df_admitsFYI.copy()
    feat_imp_df = feat_imp_df.drop(columns=['UG College', 'UG Major','Year'])
    feat_imp_df['Season'] = feat_imp_df['Season'].map({'Fall': 0, 'Spring': 1})
    major_encoder = LabelEncoder()
    major_encoder.fit(feat_imp_df['Target Major'])
    feat_imp_df['Target Major'] = major_encoder.transform(feat_imp_df['Target Major'])
    X = feat_imp_df.iloc[:, 2:]
    major_feat_imp = pd.DataFrame(suggest_uni_model.feature_importances_, index=X.columns, columns=['Importance']).sort_values(by='Importance', ascending=False)

    # Assuming you have the same data as before
    std_devs = np.std([tree.feature_importances_ for tree in suggest_uni_model.estimators_], axis=0)
    feature_importances = {'Feature': major_feat_imp.index, 'Importance': major_feat_imp['Importance']}

    # Creating the Plotly bar plot
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=feature_importances['Feature'],
        y=feature_importances['Importance'],
        error_y=dict(
            type='data',
            array=std_devs,
            visible=True
        ),
        marker_color='#1f77b4',
        name='Feature Importances'
    ))

    # Customizing the layout
    fig.update_layout(
        title='Features and their importance',
        xaxis_title='Features',
        yaxis_title='Importance',
        bargap=0.1
    )

    # Displaying the plot
    return fig