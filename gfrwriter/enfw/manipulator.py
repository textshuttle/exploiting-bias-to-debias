#!/usr/bin/env python3

# Parts of code taken from Sun et al. (2021) https://github.com/googleinterns/they-them-theirs

import re
import random

from typing import Dict, List

from collections import defaultdict

from gfrwriter.en.manipulator import EnglishManipulator, \
                                     EnglishRuleBasedReverser, \
                                     EnglishNormalizer, \
                                     PRONOUNS_MALE, PRONOUNS_FEMALE, NOUNS_MALE, \
                                     NOUNS_FEMALE, IRREGULAR_VERBS


import spacy
from spacy.lang.en import English
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, NOUN

random.seed(2022)

def reverse_dict(dictionary: Dict) -> Dict:
    rev_dict = dict()
    for k, v in dictionary.items():
        rev_dict[v] = k
    return rev_dict

FORMS_TO_CHANGE = {'he': 'she', 'him': 'her', 'his': 'her', 'himself': 'herself',
                 'she': 'he', 'her': 'him', 'hers': 'his', 'herself': 'himself'}

PRONOUNS_MALE = reverse_dict(PRONOUNS_MALE)

PRONOUNS_FEMALE = reverse_dict(PRONOUNS_FEMALE)

NOUNS_MALE = reverse_dict(NOUNS_MALE)

NOUNS_FEMALE = reverse_dict(NOUNS_FEMALE)

IRREGULAR_VERBS = reverse_dict(IRREGULAR_VERBS)


class EnglishFWManipulator(EnglishManipulator):

    def __init__(self):
        self.type = 'manipulator'

        # Nouns that are typically gendered
        self.nouns_male = self._add_upper_and_create_regex(NOUNS_MALE)
        self.nouns_female = self._add_upper_and_create_regex(NOUNS_FEMALE)

        # Pronouns that are typically gendered
        self.pronouns_male = self._add_upper_and_create_regex(PRONOUNS_MALE)
        self.pronouns_female = self._add_upper_and_create_regex(PRONOUNS_FEMALE)

        self.forms_to_change = self._add_upper_and_create_regex(FORMS_TO_CHANGE)

        nouns = [(v, None) for k,v in NOUNS_MALE.items()] + [(v, None) for k,v in NOUNS_FEMALE.items()]
        self.pronouns_to_exclude = []

    def get_reverser(delimiter: str,
                     reverse_approach: str) -> 'BaseManipulator':
        if reverse_approach == 'rule-based':
            return EnglishFWRuleBasedReverser()
        else:
            raise NotImplementedError


class EnglishFWRuleBasedReverser(EnglishFWManipulator, EnglishRuleBasedReverser):

    def __init__(self):
        super().__init__()
        self.normalizer = EnglishFWNormalizer('')
        self.spacy_tagger = spacy.load('en_core_web_lg')
        self.type = 'rule-based'

        self.re_they_have = re.compile(r'\b(They|they|THEY) have\b')
        self.re_s_he_has = re.compile(r'\b(She|she|SHE|He|he|HE) has\b')

        self.re_they_are = re.compile(r'\b(They|they|THEY) are\b')
        self.re_s_he_is = re.compile(r'\b(She|she|SHE|He|he|HE) is\b')

    def _pluralize_present_simple(self, lowercase_verb: str) -> str:
        '''
        Pluralize present simple verb forms.
        '''
        if lowercase_verb.endswith('ies'):
            return lowercase_verb[:-3] + 'y'

        if lowercase_verb.endswith('shes'):
            return lowercase_verb[:-2]

        if lowercase_verb.endswith('ches'):
            return lowercase_verb[:-2]

        if lowercase_verb.endswith('xes'):
            return lowercase_verb[:-2]

        if lowercase_verb.endswith('zes'):
            return lowercase_verb[:-2]

        if lowercase_verb.endswith('ses'):
            return lowercase_verb[:-2]

        return lowercase_verb.rstrip('s')

    def _find_plural_forms(self, verb_auxiliaries: Dict) -> Dict:
        '''
        Find the plural forms of verbs that should be replaced.
        '''
        verb_replacements = dict()

        for verb, auxiliaries in verb_auxiliaries.items():
            if 'Number=Plur' in verb.morph:
                verb_replacements[verb] = None
                continue

            if not auxiliaries:
                if 'Tense=Past' in verb.morph:
                    # Handle past tense of "to be"
                    if verb.text.lower() == 'was':
                        verb_replacements[verb] = self._capitalize(verb.text, 'were')
                    # Other past-tense verbs remain the same
                    else:
                        verb_replacements[verb] = None

                # Oftentimes, if there are 2+ verbs in a sentence, each verb after the first (the conjuncts) will be misclassified
                # the POS of these other verbs are usually misclassified as NOUN
                # e.g. He dances and prances and sings. --> "prances" and "sings" are conjuncts marked as NOUN (should be VERB)
                # checking if verb ends with "s" is a band-aid fix
                elif 'Tense=Pres' in verb.morph or verb.text.endswith('s'):
                    if verb.text.lower() in IRREGULAR_VERBS.keys():
                        replacement = IRREGULAR_VERBS[verb.text.lower()]
                        verb_replacements[verb] = self._capitalize(verb.text, replacement)
                    else:
                        singular_form = self._pluralize_present_simple(verb.text)
                        verb_replacements[verb] = self._capitalize(verb.text, singular_form)
            else:
                verb_replacements[verb] = None  # Do not need to pluralize root verb if there are auxiliaries

                # Use a lookup to find replacements for auxiliaries
                for auxiliary in auxiliaries:
                    text = auxiliary.text
                    if text.lower() in IRREGULAR_VERBS.keys():
                        replacement = IRREGULAR_VERBS[text.lower()]
                        replacement = self._capitalize(text, replacement)
                        verb_replacements[auxiliary] = replacement
                    else:
                        verb_replacements[auxiliary] = None

        return verb_replacements

    def _pluralize_verb_forms(self, sent: 'spacy.Doc') -> str:
        '''
        Pluralize verb forms.
        '''
        verb_auxiliaries = self._find_all_verb_candidates(sent, ['he', 'she'])
        verb_replacements = self._find_plural_forms(verb_auxiliaries)

        new_toks = []
        for tok in sent:
            if tok.text.lower() == 'her':
                if tok.dep == poss:
                    new_toks.append(self._capitalize(tok.text, 'their'))
                else:
                    new_toks.append(self._capitalize(tok.text, 'them'))
            elif tok.text.lower() == 'his':
                if tok.head.pos != NOUN:
                    new_toks.append(self._capitalize(tok.text, 'theirs'))
                else:
                    new_toks.append(self._capitalize(tok.text, 'their'))

            elif tok in verb_replacements.keys() and verb_replacements[tok]:
                new_toks.append(verb_replacements[tok])
            else:
                new_toks.append(tok.text)
            if tok.whitespace_:
                new_toks.append(tok.whitespace_)

        return ''.join(new_toks)

    def change_form(self, sent: str) -> str:
        '''
        Change female forms to male forms and vice versa.
        '''
        new_toks = []
        for tok in sent:
            changed = False

            if tok.text.lower() == 'her':
                changed = True
                if tok.dep == poss:
                    new_toks.append(self._capitalize(tok.text, 'his'))
                else:
                    new_toks.append(self._capitalize(tok.text, 'him'))

            elif tok.text.lower() == 'his':
                changed = True
                if tok.head.pos != NOUN:
                    new_toks.append(self._capitalize(tok.text, 'hers'))
                else:
                    new_toks.append(self._capitalize(tok.text, 'her'))

            else:
                for r, v in self.forms_to_change.values():
                    if r.search(tok.text):
                        new_toks.append(r.sub(v, tok.text))
                        changed = True
                        break

            if not changed:
                new_toks.append(tok.text)

            if tok.whitespace_:
                new_toks.append(tok.whitespace_)

        return ''.join(new_toks)

    def reverse(self, sent: str, norm_sent: str) -> str:
        '''
        Rewrite a non gender-fair sentence to use gender-fair forms using rules.
        '''
        if not self.contains_gendered_form(sent):
            return sent, sent

        tagged = self.spacy_tagger(sent)
        plural_verb_sent = self._pluralize_verb_forms(tagged)

        other_form = self.change_form(tagged)

        gender_fair_form = self._reverse_male(plural_verb_sent)
        gender_fair_form = self._reverse_female(gender_fair_form)

        # add artificial occurrences of they've
        if self.re_they_have.search(gender_fair_form):
            random_num = random.randint(1,10)
            if random_num == 1:
                gender_fair_form = self.re_they_have.sub(r"\1've", gender_fair_form)
                other_form = self.re_s_he_has.sub(r"\1's", other_form)
                sent = self.re_s_he_has.sub(r"\1's", sent)
            elif random_num == 2:
                gender_fair_form = self.re_they_have.sub(r"\1’ve", gender_fair_form)
                other_form = self.re_s_he_has.sub(r"\1’s", other_form)
                sent = self.re_s_he_has.sub(r"\1’s", sent)

        # add artificial occurrences of they're
        if self.re_they_are.search(gender_fair_form):
            random_num = random.randint(1,10)
            if random_num == 1:
                gender_fair_form = self.re_they_are.sub(r"\1're", gender_fair_form)
                other_form = self.re_s_he_is.sub(r"\1's", other_form)
                sent = self.re_s_he_is.sub(r"\1's", sent)
            elif random_num == 2:
                gender_fair_form = self.re_they_are.sub(r"\1’re", gender_fair_form)
                other_form = self.re_s_he_is.sub(r"\1’s", other_form)
                sent = self.re_s_he_is.sub(r"\1’s", sent)

        return gender_fair_form, gender_fair_form, norm_sent, other_form


class EnglishFWNormalizer(EnglishFWManipulator, EnglishNormalizer):

    def __init__(self, delimiter: str):
        super().__init__()
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 3000000

        self.to_ignore_re = re.compile(r"(he|she)('|’)s\b")
