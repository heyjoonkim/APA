from typing import List, Dict

from rouge_score import rouge_scorer

from pipeline.templates import AMBIGUOUS_PHRASES

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def analysis(dataset:List[Dict], logger) -> Dict:
    full = len(dataset)
    ## full dataset ##
    correct = len([d for d in dataset if d.get('is_correct')])

    ambiguous_dataset = [d for d in dataset if d.get('is_ambiguous')]
    num_ambiguous = len(ambiguous_dataset)
    unambiguous_dataset = [d for d in dataset if not d.get('is_ambiguous')]
    num_unambiguous = len(unambiguous_dataset)

    ## ambiguous ##
    ambiguous_correct = len([d for d in ambiguous_dataset if d.get('is_correct')])

    ## unambiguous ##
    unambiguous_correct = len([d for d in unambiguous_dataset if d.get('is_correct')])
    unambiguous_clarified = len([d for d in unambiguous_dataset if d.get('is_ambiguous_prediction') and d.get('is_correct') is False])

    ## accuracy ##
    accuracy = correct / full
    ambiguous_accuracy = ambiguous_correct / num_ambiguous
    unambiguous_accuracy = unambiguous_correct / num_unambiguous
    unambiguous_clarified_accuracy = unambiguous_clarified / num_unambiguous
    
    if ambiguous_accuracy + unambiguous_accuracy == 0:
        hscore = -100
    else:
        hscore = 2 * ambiguous_accuracy * unambiguous_accuracy / (ambiguous_accuracy + unambiguous_accuracy)
        
    ## New metric: precision - recall - F1 for ambiguous samples only ##
    num_clarified = unambiguous_clarified + ambiguous_correct
    
    precision = 0 if num_clarified == 0 else ambiguous_correct / num_clarified 
    recall = ambiguous_accuracy
    f1 = 0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    logger.info(f'EVALUTAION RESULTS:')
    logger.info(f'Unambiguous accuracy               (correct answer / total unambig)   : {round(unambiguous_accuracy*100, 2):<2}% ({unambiguous_correct} / {num_unambiguous})')
    logger.info(f'Unambiguous Clarification accuracy (clarification / total unambig)    : {round(unambiguous_clarified_accuracy*100, 2):<2}% ({unambiguous_clarified} / {num_unambiguous})')
    logger.info('===============================================================================')
    logger.info(f'Ambiguous accuracy (= recall)      (clarification / total ambig)      : {round(ambiguous_accuracy*100, 2):<2}% ({ambiguous_correct} / {num_ambiguous})')
    logger.info('===============================================================================')
    logger.info(f'Total accuracy                                                        : {round(accuracy*100, 2):<2}% ({correct} / {full})')
    logger.info(f'* H-score (UniDA)                                                     : {round(hscore*100, 2):<2}')
    logger.info('===============================================================================')
    logger.info(f'Ambiguous Precision                (correct ambig / ambig prediction) : {round(precision*100, 2):<2}% ({ambiguous_correct} / {num_clarified})')
    logger.info(f'Ambiguous Recall                   (correct ambig / total ambig)      : {round(recall*100, 2):<2}% ({ambiguous_correct} / {num_ambiguous})')
    logger.info(f'* Ambiguous F1                                                        : {round(f1*100, 2):<2}')
    
    return dict(
        accuracy=accuracy,
        ambiguous_accuracy=ambiguous_accuracy,
        unambiguous_accuracy=unambiguous_accuracy,
    )


    

def rouge_eval(prediction:str, answers:List[str], is_ambiguous:bool, threshold:float=0.3, analysis:bool=True, **kwargs) -> Dict:
    is_ambiguous_prediction = None
    
    prediction = prediction.strip().lower()
    
    if is_ambiguous:
        ## Evaluate ambiguous samples
        contains_ambig = sum([int(ambiguous_phrase in prediction) for ambiguous_phrase in AMBIGUOUS_PHRASES])
        is_correct = True if contains_ambig > 0 else False
        eval_score = None
    else:
        ## Evaluate unambiguous samples    
        rouge_scores = [scorer.score(prediction, answer.lower()).get('rougeL').fmeasure for answer in answers]
        
        eval_score = max(max(rouge_scores), 0.0)

        is_correct = bool(eval_score > threshold)
    
    if analysis:
        # check if the unambiguous prediction contains ambiguous phrases
        contains_ambig = sum([int(ambiguous_phrase in prediction) for ambiguous_phrase in AMBIGUOUS_PHRASES])
        is_ambiguous_prediction = True if contains_ambig > 0 else False
    
    return dict(
        score=eval_score,                                       # rougeL score (for unambiguous samples, None for ambiguous samples)
        is_correct=is_correct,                                  # if predicted GT answer = True
        is_ambiguous_prediction=is_ambiguous_prediction,        # if predicted as ambiguous = True
    )
        

def get_evaluation_method(evaluation_method:str):
    if evaluation_method == 'rouge':
        return rouge_eval
    else:
        raise NotImplementedError(f'evaluation_method {evaluation_method} not implemented.')