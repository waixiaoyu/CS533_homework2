# -*- coding: utf-8 -*-

import nltk
import vocabulary
import itertools
import numpy as np
import scipy
import sklearn
import heapq
import tagtools
from TaggingExperiment import *

reference_train_file, reference_test_file, reference_dev_file = \
  "reference_train.xml", "reference_test.xml", "reference_dev.xml"
reference_xml_item_keyword = "entry"

ingredients_train_file, ingredients_test_file, ingredients_dev_file = \
  "ingredients_small_train.xml", "ingredients_small_test.xml", "ingredients_devset.xml"
#  "ingredients_big_train.xml", "ingredients_big_test.xml", "ingredients_devset.xml"
ingredients_xml_item_keyword = "ingredient"

def tagged_contexts(seq) :
    '''take the tagged tokens in seq and create a new sequence 
    that yields the same tokens but presents them with the full
    context that precedes and follows them'''
    items = [x for x in seq]
    words = [w for (w,_) in items]
    for i, (w, t) in enumerate(items) :
        if i == 0 :
            before = []
        else :
            before = words[i-1::-1]
        after = words[i+1:]
        yield (w, before, after), t
        
        
def make_cxt_feature_processor(word_tests, list_tests) :
    def feature_processor(features, item) :
        this, before, after = item
        result = []
        
        def addf(name) :
            if name :
                r = features.add(name)
                if r :
                    result.append(r)

        def add_word_features(w, code) :
            for ff in word_tests :
                addf(ff(w, code))

        def add_list_features(l, code) :
            for ff in list_tests :
                addf(ff(l, code))

        first = lambda l: l[0] if l else None
        
        add_word_features(this, u"w")
        add_list_features(before, u"-")
        add_list_features(after, u"+")
        for wx, cx in [(first(before), u"-w"),
                       (first(after), u"+w")] :
            if wx :
                add_word_features(wx, cx)
    
        return np.array(result)
    return feature_processor


identity_feature = lambda item, code: "{}: {}".format(code, item)

def all_digits(item, code) :
    if item.isdigit() :
        return u"{}: is all digits".format(code)
    
def lonely_initial(item, code) :
    if len(item) == 2 and item[0].isupper and item[1] == '.' :
        return u"{}: lonely initial".format(code)
    
def is_empty(l, code) :
    if not l :
        return u"{}: empty".format(code)
    
    
default_tokenizer = \
    lambda i: tagged_contexts(tagtools.bies_tagged_tokens(i))
default_token_view = lambda i : i[0]
default_feature_processor = \
    make_cxt_feature_processor([all_digits, lonely_initial, 
                                identity_feature],
                               [is_empty])
def default_features(vocab) :
    return lambda data: vocab

bib_features = vocabulary.Vocabulary()

bib_data = tagtools.DataManager(reference_train_file, 
                                reference_test_file, 
                                reference_dev_file,
                                reference_xml_item_keyword,
                                default_tokenizer,
                                default_token_view,
                                default_features(bib_features),
                                default_feature_processor)

bib_data.initialize()

#bib_data.test_features_dev_item(0)







#==============================================================================
#build tag-tag prob 
#==============================================================================

def getTagProb(items):
    for i in range(len(items)):
        for pair in tagtools.bies_tagged_tokens(items[i]):
            yield pair

def getTagExpt(data):
    expt1 = Experiment()
    expt1.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams((tag for (word, tag) in getTagProb(data.train_items))))
    expt1.cpd_tags = nltk.ConditionalProbDist(expt1.cfd_tags, nltk.MLEProbDist)
    expt1.tagset = set((tag for (word, tag) in \
                        getTagProb(data.train_items)))
    return expt1

class Experiment(object) :
    pass

expt1=getTagExpt(bib_data)

#==============================================================================
# 
#==============================================================================

# a good default
def initial_status():
    return frozenset(['START'])

# terrible.  work out the right answer for yourself
def everything_is_consistent(t1, t2) :
    if t1=='START' or t2=='END' or expt1.cpd_tags[t1].prob(t2)>0.01 :
        return True
    else:
        return False
    

# doesn't enforce unique specification.  sometimes you should.
def status_whats_seen(status, t1, t2) :
    field = t2[3:]
    return status.union([field])

# enforce model constraints all the time.  you can do better.
def dont_do_special(status, t1, j, node, heuristics):
    return []





#classifier and decoder    

def classifierAndDecoder(te):
    te.initialize()
    te.fit_and_validate()
    print te.accuracy
    te.visualize_classifier(10)
    te.decode_and_validate()
    
    print sum([1 for t in te.dev_partials if t]), "correct with omissions"
    print sum([1 for t in te.dev_exacts if t]), "fully correct"
    print tagtools.bieso_classification_report(te.dev_y, te.dev_predictions)
    #
    print tagtools.bieso_classification_report(te.dev_y, te.dev_decoded)

#==============================================================================
# bib
#==============================================================================

bib_classifier = sklearn.linear_model.SGDClassifier(loss="log",
                                           penalty="elasticnet",
                                           n_iter=5)

bib_decoder = tagtools.BeamDecoder(initial_status,
                                   everything_is_consistent,
                                   status_whats_seen,
                                   dont_do_special)

bib = TaggingExperiment(bib_data, 
                        bib_features,
                        bib_classifier,
                        bib_decoder)
classifierAndDecoder(bib)


#==============================================================================
# recipe
#==============================================================================

recipe_features = vocabulary.Vocabulary()

recipe_data = tagtools.DataManager(ingredients_train_file, 
                                   ingredients_test_file,
                                   ingredients_dev_file,
                                   ingredients_xml_item_keyword,
                                   default_tokenizer,
                                   default_token_view,
                                   default_features(recipe_features),
                                   default_feature_processor)

recipe_data.initialize()

expt1=getTagExpt(recipe_data)

#recipe_data.test_features_dev_item(0)





recipe_classifier = sklearn.linear_model.SGDClassifier(loss="log",
                                           penalty="elasticnet",
                                           n_iter=5)

recipe_decoder = tagtools.BeamDecoder(initial_status,
                                   everything_is_consistent,
                                   status_whats_seen,
                                   dont_do_special)

recipe = TaggingExperiment(recipe_data, 
                        recipe_features,
                        recipe_classifier,
                        recipe_decoder)
classifierAndDecoder(recipe)
    
