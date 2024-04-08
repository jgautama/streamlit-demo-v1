from models.model_university import * #get_top_five_universities

"""
purpose: list of universities name that we offered to be predicted
return: list of universities name (string)
"""
def _get_list_universities_name():
    university_names = get_top_five_universities()
    return university_names


"""
purpose: retrieve the list of majors offered based on the university name
return: list of majors (string)
"""
def _get_major_by_university(university_name: str):
    majors = get_top_majors_by_university(university_name)
    return majors
    # print("university name: ", university_name)
    # if university_name == "University of Washington":
    #     return ["Informatics", "CSE", "Data Science"]
    # else:
    #     return ["None"]


def _get_all_universities_majors():
    return get_list_majors_per_universities()

"""
purpose: normalize the test scores from 0 - 9.
return: normalized score from 0 to 9 (integer)
"""
def _normalize_toefl_ielts_score(score: int) -> int:
    normalized_score = 0
    
    if score <= 9:  #IELTS
        normalized_score = round((score / 9) * 9, 2)
    else:  #TOEFL
        normalized_score = round((score / 120) * 9, 2)
        
    return normalized_score

def _predict_acceptance(map, university_major):
    return 1 #get_acceptance_score(map, university_major)

"""
purpose: retrieve all universities name from the dataset
return: list of universities name (string)
"""
def _get_all_universities():
    return get_universities()

"""
purpose: retrieve all majors name from the dataset
return: list of majors name (string)
"""
def _get_major_by_university():
    return get_uni_major()

"""
purpose: Get Average Scores for particular university and major
return: dictionary of average scores
"""
def _get_applied_info(university_name, major_name):
    return get_applied_info(university_name, major_name)