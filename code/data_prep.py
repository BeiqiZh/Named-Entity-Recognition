
import pandas as pd
from tqdm.notebook import tqdm
import spacy
from spacy.tokens import DocBin

def search_keyword_loc_old(texts,dict_keywords_label):
    '''
    This Function is used to:
    Locate the keyword we want
    dict_indext: The dictionary including the keyword and their corresponding label
    return: the list with input text data with location of keywords(start and end position)
    '''
    list_res=[]
    for text in texts:
        dict_index={"entity":list()}
        for i in range(len(dict_keywords_label)):

            for key, label in dict_keywords_label[i].items():
                if text.find(key)!=-1:
                    dict_index["entity"].append((text.find(key),
                                                    (len(key)+text.find(key)),   label))

        list_res.append((text,dict_index))
    return list_res



def filter_spans(spans):
#  """Filter a sequence of spans and remove duplicates or overlaps. Useful for
#  creating named entities (where one token can only be part of one entity) or
#  when merging spans with `Retokenizer.merge`. When spans overlap, the (first)
#  longest span is preferred over shorter spans.

#  spans (iterable): The spans to filter.
#  RETURNS (list): The filtered spans.
#  """
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
     # Check for end - 1 here because boundaries are inclusive
     if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
     # Check for end - 1 here because boundaries are inclusive
     if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result


def convert_to_spacy(data,path="/Users/fandi/Desktop/deep_learning/corups/train.spacy"):
    nlp = spacy.blank("en") # load a new spacy model
    db = DocBin() # create a DocBin object
    for text, annot in tqdm(data): # data in previous format
        doc = nlp.make_doc(text) # create doc object from text
        ents = []
        
        for start, end, label in annot["entities"]: # add character indexes
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            if span is None:
                print("Skipping entity")
            else:
                ents.append(span)
        ents = filter_spans(ents)
        doc.set_ents(ents) # label the text with the ents
        db.add(doc)

    db.to_disk(path) # save the docbin object