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