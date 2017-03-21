# -*- coding: utf-8 -*-

import nltk
import vocabulary
import itertools
import numpy as np
import scipy
import sklearn
import heapq
import tagtools

reference_train_file, reference_test_file, reference_dev_file = \
  "reference_train.xml", "reference_test.xml", "reference_dev.xml"
reference_xml_item_keyword = "entry"

ingredients_train_file, ingredients_test_file, ingredients_dev_file = \
  "ingredients_small_train.xml", "ingredients_small_test.xml", "ingredients_devset.xml"
#  "ingredients_big_train.xml", "ingredients_big_test.xml", "ingredients_devset.xml"
ingredients_xml_item_keyword = "ingredient"

#==============================================================================
# TaggingExperiment
#==============================================================================

class TaggingExperiment(object) :
    '''Organize the process of getting data, building a classifier,
    and exploring new representations'''
    
    def __init__(self, data, features, classifier, decoder) :
        'set up the problem of learning a classifier from a data manager'
        self.data = data
        self.classifier = classifier
        self.features = features
        self.decoder = decoder
        self.initialized = False
        self.trained = False
        self.decoded = False
        
    def initialize(self) :
        'materialize the training data, dev data and test data as matrices'
        if not self.initialized :
            self.train_X, self.train_y, self.train_d = self.data.training_data()
            self.features.stop_growth()
            self.dev_X, self.dev_y, self.dev_d = self.data.dev_data()
            self.test_X, self.test_y, self.test_d = self.data.test_data()
            self.initialized = True
        
    def fit_and_validate(self) :
        'train the classifier and assess predictions on dev data'
        if not self.initialized :
            self.initialize()
        self.classifier.fit(self.train_X, self.train_y)
        self.tagset = self.classifier.classes_
        self.trained = True
        self.dev_predictions = self.classifier.predict(self.dev_X)
        self.accuracy = sklearn.metrics.accuracy_score(self.dev_y, self.dev_predictions)
    
    def visualize_classifier(self, item_number) :
        'show the results of running the classifier on text number item_number'
        if not self.trained :
            self.fit_and_validate()
        w = self.data.dev_item_token_views(item_number)
        s = self.dev_d[item_number]
        e = self.dev_d[item_number+1]
        tagtools.visualize(w, {'actual': self.dev_y[s:e], 
                               'predicted': self.dev_predictions[s:e]})

    def decode_and_validate(self) :
        '''use the trained classifier and beam search to find the consistent
        analyses of all the items in the dev data'''
        if not self.trained :
            self.fit_and_validate()
        self.dev_log_probs = self.classifier.predict_log_proba(self.dev_X)
        results = []
        self.dev_partials = []
        self.dev_exacts = []
        for i in range(len(self.dev_d)-1) :
            s = self.dev_d[i]
            e = self.dev_d[i+1]
            tags, score = self.decoder.search(self.tagset, self.dev_log_probs[s:e])
            p_t = tags[1:-1]
            results.append(p_t)
            self.dev_partials.append(tagtools.agrees(p_t, iter(self.dev_y[s:e]), partial=True))
            self.dev_exacts.append(tagtools.agrees(p_t, iter(self.dev_y[s:e]), partial=False))
        self.dev_decoded = np.concatenate(results)

    def visualize_decoder(self, item) :
        'show the results of running the classifier and decoder on text number item_number'
        if not self.decoded :
            self.decode_and_validate()
        w = self.data.dev_item_token_views(item)
        s = self.dev_d[item]
        e = self.dev_d[item+1]
        tagtools.visualize(w, {'actual': self.dev_y[s:e], 
                               'best': self.dev_predictions[s:e],
                               'predicted': self.dev_decoded[s:e]})

    @classmethod
    def transform(cls, expt, operation, classifier) :
        'use operation to transform the data from expt and set up new classifier'
        if not expt.initialized :
            expt.initialize()
        result = cls(expt.data, classifier)
        result.train_X, result.train_y, result.train_d = \
            operation(expt.train_X, expt.train_y, expt.train_d)
        result.dev_X, result.dev_y, result.dev_d = \
            operation(expt.dev_X, expt.dev_y, expt.dev_d)
        result.test_X, result.test_y, result.test_d = \
            operation(expt.test_X, expt.test_y, expt.test_d)
        result.initialized = True
        return result



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

def all_digits_bib(item, code) :
    if item.isdigit() :
        if len(item)==4:
            return u"{}: is year".format(code)
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
    make_cxt_feature_processor([all_digits_bib, lonely_initial, 
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

bib_data.test_features_dev_item(0)







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
    if t1=='START' or t2=='END' or expt1.cpd_tags[t1].prob(t2)>0.001 :
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


def status_whats_seen_bib(status, t1, t2) :
    field = t2[3:]
    if field=='booktitle' and status.issuperset(['journal']):
        return None
    if field=='journal' and status.issuperset(['booktitle']):
        return None
    
    if t2[0]=='s' and status.issuperset([field]):
        return None
    else:
        return status.union([field])
    

bib_decoder = tagtools.BeamDecoder(initial_status,
                                   everything_is_consistent,
                                   status_whats_seen_bib,
                                   dont_do_special)

bib = TaggingExperiment(bib_data, 
                        bib_features,
                        bib_classifier,
                        bib_decoder)
classifierAndDecoder(bib)


#==============================================================================
# recipe
#==============================================================================

#recipe_features = vocabulary.Vocabulary()
#
#recipe_data = tagtools.DataManager(ingredients_train_file, 
#                                   ingredients_test_file,
#                                   ingredients_dev_file,
#                                   ingredients_xml_item_keyword,
#                                   default_tokenizer,
#                                   default_token_view,
#                                   default_features(recipe_features),
#                                   default_feature_processor)
#
#recipe_data.initialize()
#
#expt1=getTagExpt(recipe_data)
#
##recipe_data.test_features_dev_item(0)
#
#
#
#def status_whats_seen_recipe(status, t1, t2) :
#    field = t2[3:]
#    if field=='comment' or field=='other':
#        return status.union([field])
#    
#    if t2[0]=='s' and status.issuperset([field]) :
#        return None
#    else:
#        return status.union([field])
#
#recipe_classifier = sklearn.linear_model.SGDClassifier(loss="log",
#                                           penalty="elasticnet",
#                                           n_iter=5)
#
#recipe_decoder = tagtools.BeamDecoder(initial_status,
#                                   everything_is_consistent,
#                                   status_whats_seen_recipe,
#                                   dont_do_special)
#
#recipe = TaggingExperiment(recipe_data, 
#                        recipe_features,
#                        recipe_classifier,
#                        recipe_decoder)
#classifierAndDecoder(recipe)
    
