import sys
# setting path
sys.path.append('..')
# importing
from src import metrics

def test_metrics():
    """Test the metrics.py file functions with printout
    
    Args:
        None 
    
    Returns:
        None
    """
    #Test Case for multiple captions
    candidate= ["a man is speaking", "rain falls"]
    reference = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]
    #Test Case for single caption
    #reference = ['this is small test']
    #candidate = ['this is a test']
    #Test Cider score with printout
    print(metrics.calculate_meteor_score_sentence(reference, candidate))