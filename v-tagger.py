# -*- coding: utf-8 -*-


# coding: utf-8

# This notebook walks through the creation of an HMM bigram part of speech tagger using NLTK and explains the Viterbi algorithm for finding the most likely tag sequence for a sentence.
# It's adapted by Matthew Stone for CS 533 from Chapter 5 of the Bird, Klein & Loper's NLTK book and by course notes by Katrin Erk.

# In[1]:

import nltk
from nltk.corpus import brown


# Estimating $P(w_i | c_i)$ from corpus data using Maximum Likelihood Estimation (MLE):
# $$P(w_i | c_i) = \frac{count(w_i, c_i)}{count(c_i)}$$
# 
# We add an artificial `START` tag at the beginning of each sentence, and
# we add an artificial `END` tag at the end of each sentence.
# So we start out with the brown tagged sentences,
# and make a generator that goes through each sentence, giving the initial dummy tag,
# the real tags, and then the final dummy tag.  

# In[2]:

def brown_sentence_items() :
    for sent in brown.tagged_sents(tagset='universal') :
        yield ('START', 'START')
        for (word, tag) in sent :
            yield (tag, word)
        yield ('END', 'END')


# Then you can use built-in NLTK tools to get the probability distributions you need.

# In[3]:

class Experiment(object) :
    pass

expt1 = Experiment()
expt1.cfd_tagwords = nltk.ConditionalFreqDist(brown_sentence_items())
expt1.cpd_tagwords = nltk.ConditionalProbDist(expt1.cfd_tagwords, nltk.MLEProbDist)


# In[4]:

print "The probability of an adjective (NOUN) being 'huge' is",     expt1.cpd_tagwords["NOUN"].prob("huge")



# Estimating $P(c_i | c_{i-1})$ from corpus data using Maximum Likelihood Estimation (MLE):
# $$P(c_i | c_{i-1}) = \frac{count(c_{i-1}, c_i)}{count(c_{i-1})}$$
# Similar story: make the generator for the token sequence and use NLTK tools.

# In[5]:

expt1.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams((tag for (tag, word) in brown_sentence_items())))
expt1.cpd_tags = nltk.ConditionalProbDist(expt1.cfd_tags, nltk.MLEProbDist)
expt1.tagset = set((tag for (tag, word) in brown_sentence_items()))


# In[6]:

print "If we have just seen 'DET', the probability of 'X' is",    expt1.cpd_tags["DET"].prob("X")


# Larger test: what is the probability of the tag sequence "PRON VERB PRT VERB" for the word sequence "I want to race"?  It is
# $$\begin{array}{ccc} P(START) * & P(PRON|START) * & P(I | PRON) * \\
#           & P(VERB | PRON) * & P(want | VERB) * \\
#            & P(PRT | VERB) * & P(to | PRT) * \\
#            & P(VERB | PRT) * & P(race | VERB) * \\
#             & & P(END | VERB)\end{array}$$
#             

# In[7]:

prob_tagsequence = expt1.cpd_tags["START"].prob("PRON") * expt1.cpd_tagwords["PRON"].prob("I") *     expt1.cpd_tags["PRON"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob("want") *     expt1.cpd_tags["VERB"].prob("PRT") * expt1.cpd_tagwords["PRT"].prob("to") *     expt1.cpd_tags["PRT"].prob("VERB") * expt1.cpd_tagwords["VERB"].prob("race") *     expt1.cpd_tags["VERB"].prob("END")

print "The probability of the tag sequence 'START PRON VERB PRT VERB END' for 'I want to race' is:", prob_tagsequence


# Here we walk through the Viterbi algorithm, a dynamic programming procedure for finding the optimal tag sequence.  The main idea is that we're incrementally computing the best tag sequences up to a specific word, except that to reason correctly, we need to keep track of all the possible different tags that word could have.  
# 
# We initialize with the first word.  This gives us a table saying how likely the word is to have each of the tags in our tag set.

# In[8]:

def viterbi_init(expt, word) :
    prob = {}
    back = {}
    for tag in expt.tagset :
        if tag == 'START' :
            continue
        prob[ tag ] = expt.cpd_tags['START'].prob(tag) * expt.cpd_tagwords[tag].prob( word )
        back[ tag ] = 'START'
    return (prob, back)


# Now we assume we have a table of previous results.  That table tells us, for each tag in our tag set, what the probability is of the best tag sequence that covers the whole sentence up to the previous word and that assigns that previous word the target tag.  If we know that, to find the best path up to the current word, we can consider all the possible transitions from the last tag to the current tag.  The probability estimates also take into account the probability of seeing the word.

# In[9]:

def viterbi_extend(expt, prev_prob, word) :
    prob = {}
    back = {}
    for tag in expt.tagset :
        if tag == 'START' :
            continue
        best_previous = max(prev_prob.keys(), 
                            key = lambda prevtag: \
                            prev_prob[prevtag] * expt.cpd_tags[prevtag].prob(tag))
        prob[tag] = prev_prob[best_previous] * expt.cpd_tags[best_previous].prob(tag) * expt.cpd_tagwords[tag].prob(word)
        back[tag] = best_previous
    return (prob, back)


# So to analyze the whole sequence, we initialize with the first word and then extend the table step by step until we reach the end of the sentence.

# In[10]:

def viterbi_run(expt, sentence) :
    (prob, back) = viterbi_init(expt, sentence[0])
    history = [back]
    for i in range(1, len(sentence)) :
        (prob, back) = viterbi_extend(expt, prob, sentence[i])
        history.append(back)
    return (prob, history)


# At that point, we can find the overall best entry, and then trace back to find the overall tagging that this entry is derived from.

# In[11]:

def viterbi_decode(expt, prob, history, sentence) :
        best_previous = max(prob.keys(), 
                            key = lambda prevtag: \
                            prob[prevtag] * expt.cpd_tags[prevtag].prob('END'))
        p = prob[best_previous] * expt.cpd_tags[best_previous].prob('END')
        tags = [ 'END', best_previous ]
        history.reverse()
        current_best_tag = best_previous
        for bp in history:
            tags.append(bp[current_best_tag])
            current_best_tag = bp[current_best_tag]
        tags.reverse()
        return (p, zip(tags, ['START'] + sentence + ['END']))
    
def viterbi(expt, sentence) :
    prob, history = viterbi_run(expt, sentence)
    return viterbi_decode(expt, prob, history, sentence)


# In[12]:

viterbi(expt1, ['I', 'want', 'to', 'race'])


# Looking ahead, we might want to see whether the tag set has any influence in the way the model represents linguistic dependencies...

# In[13]:

def real_brown_sentence_items() :
    for sent in brown.tagged_sents() :
        yield ('START', 'START')
        for (word, tag) in sent :
            yield (tag, word)
        yield ('END', 'END')
expt2 = Experiment()
expt2.cfd_tagwords = nltk.ConditionalFreqDist(real_brown_sentence_items())
expt2.cpd_tagwords = nltk.ConditionalProbDist(expt2.cfd_tagwords, nltk.MLEProbDist)
expt2.cfd_tags = nltk.ConditionalFreqDist(nltk.bigrams((tag for (tag, word) in real_brown_sentence_items())))
expt2.cpd_tags = nltk.ConditionalProbDist(expt2.cfd_tags, nltk.MLEProbDist)
expt2.tagset = set((tag for (tag, word) in real_brown_sentence_items()))


# The answer is not going to be simple: it's going to involve learning something about English syntax and word order...

# In[14]:

viterbi(expt2, ['I', 'want', 'to', 'race'])


# In[ ]:


