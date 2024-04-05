from typing import List, Optional, Union
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
import subprocess
import evaluate

def calculate_score(outputs: List[dict], filepath: str, uuid: str)->dict:
    resFile=f'./results/run/validation_preds_{uuid}.json'
    with open(resFile, 'w') as f:
        json.dump(outputs, f)

    with open(filepath, 'a') as f:
        f.write("\n\n")
        f.write(json.dumps(outputs))
        
    annFile = './data/MSRVTT/annotations/MSR_VTT.json'
    coco = COCO(annFile)
        
    cocoRes = coco.loadRes(resFile)
    cocoEval = COCOEvalCap(coco, cocoRes)
    cocoEval.params['image_id'] = cocoRes.getImgIds()
    cocoEval.evaluate()
    out = {}
    for metric, score in cocoEval.eval.items():
        out[metric] = score*100
        print(f"{metric}: {score*100}")

    with open(filepath, 'a') as f:
        f.write("\n\n")
        f.write(json.dumps(out))
    

def calculate_bleu_score_corpus(references:List[List[str]], candidates:List[str])->float:
    """Calculate the BLEU-4 score for a corpus of sentences, that is multiple candidates.
    
    Args:
        references: A list containing a list of correct captions for each image i.e [['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]
        candidates: A list containing the predicted captions for each image i.e ['The cat is on the mat','The cat is not on the mat']

    Returns:
        float: The BLEU-4 score for the entire corpus
    """    
    #Asserts to ensure the passed parameters follow the correct format
    assert len(references) == len(candidates), "The lengths of references and candidates must be the same"
    assert isinstance(references,list), "References must be a list as it is looking at multiple captions"
    assert isinstance(candidates,list), "Candidates must be a list as it is looking at multiple captions"

    #Tokenize the sentences
    for i in references:
        if isinstance(i, list):
            for j in i:
                j=word_tokenize(j)
        else:
            i=[word_tokenize(i)]
    for i in candidates:
        i=word_tokenize(i)

    #https://www.nltk.org/api/nltk.translate.bleu_score.html
    return corpus_bleu(references, candidates)*100

























'''from typing import List, Optional, Union

# NLTK is standard and comes with python as standard library
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
#nltk.download('wordnet')
#nltk.download('punkt')

# Evaluate also has all BLEU and METEOR scores if standard nltk doesn't work, right now used for Rouge-L
import evaluate

# aac-metrics is a package that has the CIDEr-D score
# from aac_metrics.functional import cider_d
# from aac_metrics.utils.tokenization import preprocess_mono_sents, preprocess_mult_sents
import subprocess

def calculate_bleu_score_sentence(reference:Union[str,List[str]], candidate:Union[str,List[str]])->float:
    """This function calculates the BLEU-4 score for a single image caption prediction
    
    Args:
        reference: A list containing the correct captions for an image i.e ['The cat is on the mat', 'there is a cat on the wall']
        candidate: A list containing the predicted caption for an image i.e ['The cat is on the mat']
    
    Returns: The BLEU-4 score for the sentence
    """    

    #Tokenizes the reference and takes care of the two different input formats
    if isinstance(reference, list):
        reference=[word_tokenize(i) for i in reference]
    else:
        reference=[word_tokenize(reference)]
    
    #Tokenizes the candidate and takes care of the two different input formats
    if isinstance(candidate, list):
        candidate=word_tokenize(candidate[0])
    else:
        candidate=word_tokenize(candidate)
    
    #https://www.nltk.org/api/nltk.translate.bleu_score.html
    return sentence_bleu(reference, candidate)*100


def calculate_meteor_score_sentence(reference:Union[str,List[str]], candidate:Union[str,List[str]])->float:
    """This function calculates the METEOR score for a single image caption prediction

    Args:
        reference: A list containing the correct captions for an image i.e ['The cat is on the mat', 'there is a cat on the wall']
        candidate: A list containing the predicted caption for an image i.e ['The cat is on the mat']

    Returns:
        float: The METEOR score for the sentence
    """
    
    #Tokenizes the reference and takes care of the two different input formats
    if isinstance(reference, list):
        reference=[word_tokenize(i) for i in reference]
    else:
        reference=[word_tokenize(reference)]

    #Tokenizes the candidate and takes care of the two different input formats
    if isinstance(candidate, list):
        candidate=word_tokenize(candidate[0])
    else:
        candidate=word_tokenize(candidate)
    
    #https://hyperskill.org/learn/step/31112
    return meteor_score(reference, candidate)*100


def calculate_meteor_score_corpus(references:List[List[str]], candidates:List[str])->float:
    """This function calculates the METEOR score for a corpus of sentences, that is multiple candidates.

    Args:
        references: A list containing a list of correct captions for each image i.e [['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]
        candidates: A list containing the predicted captions for each image i.e ['The cat is on the mat','The cat is not on the mat']

    Returns:
        float: The METEOR score for the entire corpus
    """
    #Asserts to ensure the passed parameters follow the correct format
    assert len(references) == len(candidates), "The lengths of references and candidates must be the same"
    assert isinstance(references,list), "References must be a list as it is looking at multiple captions"
    assert isinstance(candidates,list), "Candidates must be a list as it is looking at multiple captions"

    #Take the average of the METEOR scores for the batch of captions
    meteor_total=0
    n=len(references)
    for i in range(len(references)):
        meteor_total+=(calculate_meteor_score_sentence(references[i],candidates[i]))
    return (meteor_total/n)


def calculate_rouge_score(references:List[List[str]], candidates:List[str])->float:
    """This function calculates the ROUGE-L score for a corpus of sentences, that is multiple candidates. If you need one caption, just pass a list with one element.

    Args:
        references: A list containing a list of correct captions for each image i.e [['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]
        candidates: A list containing the predicted captions for each image i.e ['The cat is on the mat','The cat is not on the mat']

    Returns:
        float: The ROUGE-L score for the entire corpus
    """
    #Loading in the route score
    rouge=evaluate.load('rouge')
    #Ensure the same amount of references and candidates
    assert len(references) == len(candidates), "The lengths of references and candidates must be the same"

    #If the reference is not a list assume its one caption and make it a list
    if not isinstance(references[0],list):
        references=[[sentence]for sentence in references]

    #If the candidate is not a list assume its one caption and make it a list
    if not isinstance(candidates,list):
        candidates=[candidates]

    #https://clementbm.github.io/theory/2021/12/23/rouge-bleu-scores.html#:~:text=In%20its%20simplest%20form%20ROUGE%20score%20is%20the,the%20denominator%20ROUGE%20is%20a%20recall%20oriented%20metric.
    return rouge.compute(predictions=candidates, references=references)['rougeL']*100



def calculate_cider_d_score(references:List[List[str]], candidates:List[str])->object:
    """This function calculates the CIDEr-D score for a corpus of sentences, that is multiple candidates. If you need one caption, just pass a list with one element.

    Args:
        references: A list containing a list of correct captions for each image i.e [['The cat is on the mat', 'there is a cat on the wall'],['The cat is not on the mat', 'there is a dog on the wall']]
        candidates: A list containing the predicted captions for each image i.e ['The cat is on the mat','The cat is not on the mat']

    Returns:
        float: The CIDEr-D score for the entire corpus
    """
    #Need to install aac-metrics packages one time
    try:
        # Using shell=True to run the command through the shell
        result = subprocess.run("aac-metrics-download", shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("Output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error occurred:", e.stderr)

    #If the reference is not a list of lists, assume its for one caption and make it a list of lists
    if not isinstance(references[0],list):
        references=[[sentence]for sentence in references]

    #CIDEr-D score only works for more than 2 captions
    assert not isinstance(candidates,str), "Must be more than 2 candidates to calculate cider-d"
    assert len(candidates)>1, "Must be more than 2 candidates to calculate cider-d"

    #Use the aac-metrics package to calculate the CIDEr-D score
    candidates = preprocess_mono_sents(candidates)
    mult_references = preprocess_mult_sents(references)
    corpus_scores, sents_scores = cider_d(candidates, mult_references)
    return corpus_scores
'''

