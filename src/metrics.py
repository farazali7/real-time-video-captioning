from typing import List, Optional, Union
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import json, os
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import sys
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu
from nltk.translate.meteor_score import meteor_score
from nltk import word_tokenize
import nltk
import subprocess
import evaluate


def calculate_score(outputs: List[dict], filepath: str, run_dir: str)->dict:
    resFile = run_dir + '/' + 'validation_preds.json'
    with open(resFile, 'w') as f:
        json.dump(outputs, f)

    with open(filepath, 'a') as f:
        f.write("\n\n")
        f.write(json.dumps(outputs))
        
    annFile = './data/MSRVTT/annotation/MSR_VTT.json'
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