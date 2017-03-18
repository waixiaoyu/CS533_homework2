# -*- coding: utf-8 -*-
import sklearn
import numpy as np
import tagtools

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