#NLTK is standard and comes with python
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
nltk.download('wordnet')
#evaluate also has all BLEU and METEOR scores if standard nltk doesn't work
import evaluate
from aac_metrics.functional import cider_d
from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
import subprocess

'''
This function calculates the BLEU-4 score for an entire corpus. That means we would have a list of all the captions in references and a list of all the predicted captions in candidates.
'''
def calculate_bleu_score_corpus(references, candidates):    
    #Example: [[ref1a sentence, ref1bsentence], [ref2a sentence, ref2b sentence]]
    #[['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]

    #Example: [cand1 sentence, cand2 sentence]
    #['The cat is on the mat','The cat is not on the mat']

    assert len(references) == len(candidates), "The lengths of references and candidates must be the same"
    assert isinstance(references,list), "References must be a list as it is looking at multiple captions"
    assert isinstance(candidates,list), "Candidates must be a list as it is looking at multiple captions"

    for i in references:
        if isinstance(i, list):
            for j in i:
                j=word_tokenize(j)
        else:
            i=[word_tokenize(i)]
    for i in candidates:
        i=word_tokenize(i)

    # Calculate the corpus_level BLEU score, weights are already defaulted to 1/4 signifying BLEU-4
    #https://www.nltk.org/api/nltk.translate.bleu_score.html
    return corpus_bleu(references, candidates)*100

'''
This function calculates the BLEU-4 score for a single sentence. That means we would have a the caption and predicted caption provided as a list. The reference is a nested list, since this function
is designed to handle multiple references that the candidate can match to and output the largest result
'''
def calculate_bleu_score_sentence(reference, candidate):    
    
    # Example: [ref1a_sentence, ref1b_sentence]
    #["The cat in the hat", "There is a cat on the mat"]
    if isinstance(reference, list):
        reference=[word_tokenize(i) for i in reference]
    else:
        reference=[word_tokenize(reference)]
    
    #Example: cand sentence
    #"The cat sits on the mat"
    if isinstance(candidate, list):
        candidate=word_tokenize(candidate[0])
    else:
        candidate=word_tokenize(candidate)
    
    # Calculate the sentence_level BLEU score same as corpus_bleu but for a single sentence
    #https://www.nltk.org/api/nltk.translate.bleu_score.html
    return sentence_bleu(reference, candidate)*100

'''
The METEOR Score could also be passed as a list of references and a candidate for a single sentence, or a list of candidates if you have multiple. It will pick the largest score.
https://hyperskill.org/learn/step/31112
'''
def calculate_meteor_score_sentence(reference, candidate):
    #Example Reference: "The cat is on the mat"
    #Example Candidate: The cat sits on the mat"
    if isinstance(reference, list):
        reference=[word_tokenize(i) for i in reference]
    else:
        reference=[word_tokenize(reference)]
        
    if isinstance(candidate, list):
        candidate=word_tokenize(candidate[0])
    else:
        candidate=word_tokenize(candidate)
    return meteor_score(reference, candidate)*100

'''
While its unclear how to do corpus METEOR score, one common suggestion is just to average for all sentences
'''
def calculate_meteor_score_corpus(references,candidates):
    #Example: [[ref1a sentence, ref1bsentence], [ref2a sentence, ref2b sentence]]
    #['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]

    #Example: [cand1 sentence, cand2 sentence]
    #['The cat is on the mat','The cat is not on the mat']

    assert len(references) == len(candidates), "The lengths of references and candidates must be the same"
    assert isinstance(references,list), "References must be a list as it is looking at multiple captions"
    assert isinstance(candidates,list), "Candidates must be a list as it is looking at multiple captions"

    meteor_scores = []
    for i in range(len(references)):
        meteor_scores.append(calculate_meteor_score_sentence(references[i],candidates[i]))
    return sum(meteor_scores)/len(meteor_scores)*100

'''
Calculates the Rouge-L score for any number of sentences
https://clementbm.github.io/theory/2021/12/23/rouge-bleu-scores.html#:~:text=In%20its%20simplest%20form%20ROUGE%20score%20is%20the,the%20denominator%20ROUGE%20is%20a%20recall%20oriented%20metric.
'''
def calculate_rouge_score(reference,candidate):
    #Example: [ref1a_sentence, ref1b_sentence] or [[ref1a_sentence, ref1b_sentence], [ref2a_sentence, ref2b_sentence]]
    #["The cat in the hat", "There is a cat on the mat"] or [["The cat in the hat", "There is a cat on the mat"], ["The cat is not on the mat", "There is a dog on the wall"]]
    #Example: cand sentence or [cand1_sentence, cand2_sentence]
    #"The cat sits on the mat" or ["The cat sits on the mat", "The cat is on the mat"]

    rouge=evaluate.load('rouge')
    assert len(reference) == len(candidate), "The lengths of references and candidates must be the same"
    if not isinstance(reference[0],list):
        reference=[[sentence]for sentence in reference]
    if not isinstance(candidate,list):
        candidate=[candidate]

    return rouge.compute(predictions=candidate, references=reference)['rougeL']*100

'''
Using AAC-metrics package to output cider-d score
'''
def calculate_cider_d_score(references,candidates):
    try:
        # Using shell=True to run the command through the shell
        result = subprocess.run("aac-metrics-download", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)
    if not isinstance(references[0],list):
        references=[[sentence]for sentence in references]
    assert not isinstance(candidates,str), "Must be more than 2 candidates to calculate cider-d"
    assert len(candidates)>1, "Must be more than 2 candidates to calculate cider-d"
    candidates = preprocess_mono_sents(candidates)
    mult_references = preprocess_mult_sents(references)
    corpus_scores, sents_scores = cider_d(candidates, mult_references)
    return corpus_scores

def test_metrics():
    candidate= ["a man is speaking", "rain falls"]
    reference = [["a man speaks.", "someone speaks.", "a man is speaking while a bird is chirping in the background"], ["rain is falling hard on a surface"]]
    #reference = ['this is small test']
    #candidate = ['this is a test']
    print(calculate_cider_d_score(reference, candidate))


#test_metrics()

