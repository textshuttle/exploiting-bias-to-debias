#!/usr/bin/env python3

# Parts of code taken from Sun et al. (2021) https://github.com/googleinterns/they-them-theirs

import re

from typing import Dict, List

from collections import defaultdict

from gfrwriter.base import BaseManipulator, \
                           RuleBasedReverser, \
                           Merger, \
                           Normalizer

import spacy
from spacy.lang.en import English
from spacy.symbols import nsubj, nsubjpass, conj, poss, obj, iobj, pobj, dobj, VERB, AUX, NOUN


TO_EXCLUDE_MF = [('he', 'she'), ('him', 'her'), ('his', 'hers'), ('himself', 'herself')]
TO_EXCLUDE_FF = [('she', 'he'), ('her', 'him'), ('hers', 'his'), ('herself', 'himself')]

PRONOUNS_MALE = {'they': 'he', 'them': 'him', 'their': 'his',
                 'theirs': 'his', 'themself': 'himself'}

PRONOUNS_FEMALE = {'they': 'she', 'them': 'her', 'their': 'her',
                   'theirs': 'hers', 'themself': 'herself'}

NOUNS_MALE = {'chairperson': 'chairman', 'chairpeople': 'chairmen',
              'anchor': 'anchorman', 'anchors': 'anchormen',
              'member of congress': 'congressman', 'members of congress': 'congressmen',
              'police officer': 'policeman', 'police officers': 'policemen',
              'spokesperson': 'spokesman', 'spokespersons': 'spokesmen',
              'flight attendant': 'steward', 'flight attendants': 'stewards',
              'principal': 'headmaster', 'principals': 'headmasters',
              'business person': 'businessman', 'business persons': 'businessmen',
              'mail carrier': 'postman', 'mail carrier': 'postmen',
              'salesperson': 'salesman', 'salespersons': 'salesmen',
              'firefighter': 'fireman', 'firefighters': 'firemen',
              'bartender': 'barman', 'bartenders': 'barmen',
              'cleaner': 'cleaning man', 'cleaners': 'cleaning men',
              'supervisor': 'foreman', 'supervisors': 'foremen',
              'average person': 'average man', 'average people': 'average men',
              'best person for the job': 'best man for the job',
              'best people for the job': 'best men for the job',
              'layperson': 'layman', 'laypeople': 'laymen',
              'husband and wife': 'man and wife', 'humankind': 'mankind',
              'human-made': 'man-made', 'skillful': 'workmanlike',
              'first-year student': 'freshman'}

NOUNS_FEMALE = {'chairperson': 'chairwoman', 'chairpeople': 'chairwomen',
                'anchor': 'anchorwoman', 'anchors': 'anchorwomen',
                'member of congress': 'congresswoman', 'members of congress': 'congresswomen',
                'police officer': 'policewoman', 'police officers': 'policewomen',
                'spokesperson': 'spokeswoman', 'spokespersons': 'spokeswomen',
                'flight attendant': 'stewardess', 'flight attendants': 'stewardesses',
                'principal': 'headmistress', 'principals': 'headmistresses',
                'business person': 'businesswoman', 'business persons': 'businesswomen',
                'mail carrier': 'postwoman', 'mail carrier': 'postwomen',
                'salesperson': 'saleswoman', 'salespersons': 'saleswomen',
                'firefighter': 'firewoman', 'firefighters': 'firewomen',
                'bartender': 'barwoman', 'bartenders': 'barwomen',
                'cleaner': 'cleaning lady', 'cleaners': 'cleaning ladies',
                'supervisor': 'forewoman', 'supervisors': 'forewomen',
                'actor': 'actress', 'actors': 'actresses',
                'hero': 'heroine', 'heroes': 'heroines',
                'comedian': 'comedienne', 'comedians': 'comediennes',
                'executor': 'executrix', 'executors': 'executrices',
                'poet': 'poetess', 'poets': 'poetesses',
                'usher': 'usherette', 'ushers': 'usherettes',
                'author': 'authoress', 'authors': 'authoresses',
                'boss': 'boss lady', 'bosses': 'boss ladies',
                'waiter': 'waitress', 'waiters': 'waitresses'}

IRREGULAR_VERBS = {"'re": "'s",
                   "’re": "’s",
                   "'ve": "'s",
                   "’ve": "’s",
                   'are': 'is',
                   'were': 'was',
                   'have': 'has',
                   'do': 'does',
                   'go': 'goes',
                   'quiz': 'quizzes'}

class EnglishManipulator(BaseManipulator):

    def __init__(self):
        self.type = 'manipulator'

        # Nouns that are typically gendered
        self.nouns_male = self._add_upper_and_create_regex(NOUNS_MALE)
        self.nouns_female = self._add_upper_and_create_regex(NOUNS_FEMALE)

        # Pronouns that are typically gendered
        self.pronouns_male = self._add_upper_and_create_regex(PRONOUNS_MALE)
        self.pronouns_female = self._add_upper_and_create_regex(PRONOUNS_FEMALE)

        nouns = [(v, None) for k,v in NOUNS_MALE.items()] + [(v, None) for k,v in NOUNS_FEMALE.items()]
        self.pronouns_to_exclude = TO_EXCLUDE_MF + TO_EXCLUDE_FF + nouns

    def _add_upper_and_create_regex(self,
                                    pairs: Dict[str, str]) -> Dict[str, str]:
        upper_pairs = {}
        for k, v in pairs.items():
            k_upper = k.upper()
            v_upper = v.upper()
            k_cap = k[0].upper()+k[1:]
            v_cap = v[0].upper()+v[1:]

            upper_pairs[k] = (re.compile(r'\b'+f'{k}'+r'\b'), v)
            upper_pairs[k_upper] = (re.compile(r'\b'+f'{k_upper}'+r'\b'), v_upper)
            upper_pairs[k_cap] = (re.compile(r'\b'+f'{k_cap}'+r'\b'), v_cap)

        return upper_pairs

    def get_reverser(delimiter: str,
                     reverse_approach: str) -> 'BaseManipulator':
        if reverse_approach == 'rule-based':
            return EnglishRuleBasedReverser()
        else:
            raise NotImplementedError

    def _contains_gendered_noun(self, sent: str) -> bool:
        for r, _ in self.nouns_male.values():
            if r.search(sent):
                return True
        for r, _ in self.nouns_female.values():
            if r.search(sent):
                return True
        return False

    def _contains_gendered_pronoun(self, sent: str)-> bool:
        for r, _ in self.pronouns_male.values():
            if r.search(sent):
                return True
        for r, _ in self.pronouns_female.values():
            if r.search(sent):
                return True
        return False

    def contains_gendered_form(self, sent: str) -> bool:
        if self._contains_gendered_pronoun(sent):
            return True
        if self._contains_gendered_noun(sent):
            return True
        return False


class EnglishRuleBasedReverser(EnglishManipulator, RuleBasedReverser):

    def __init__(self):
        super().__init__()
        self.normalizer = EnglishNormalizer('')
        self.spacy_tagger = spacy.load('en_core_web_lg')
        self.type = 'rule-based'

    def _rewrite_pronouns_male(self, sent: str) -> str:
        '''
        Rewrite gender-fair pronoun forms to generic male forms:
        e.g. they -> he
        '''
        for pronoun, (r, male_form) in self.pronouns_male.items():
            if pronoun in sent:
                sent = r.sub(male_form, sent)
        return sent

    def _rewrite_pronouns_female(self, sent: str) -> str:
        '''
        Rewrite gender-fair pronoun forms to generic female forms:
        e.g. they -> she
        '''
        for pronoun, (r, female_form) in self.pronouns_female.items():
            if pronoun in sent:
                sent = r.sub(female_form, sent)
        return sent

    def _rewrite_nouns_male(self, sent: str) -> str:
        '''
        Rewrite gender-fair nouns to generic male forms:
        e.g. firefighter -> fireman
        '''
        for noun, (r, male_form) in self.nouns_male.items():
            if noun in sent:
                sent = r.sub(male_form, sent)
        return sent

    def _rewrite_nouns_female(self, sent: str) -> str:
        '''
        Rewrite gender-fair nouns to generic male forms:
        e.g. firefighter -> firewoman
        '''
        for noun, (r, female_form) in self.nouns_female.items():
            if noun in sent:
                sent = r.sub(female_form, sent)
        return sent

    def _reverse_male(self, sent: str) -> str:
        '''
        Rewrite a gender-fair sentence to use generic male forms using rules.
        '''
        sent = self._rewrite_nouns_male(sent)
        sent = self._rewrite_pronouns_male(sent)
        return sent

    def _reverse_female(self, sent: str) -> str:
        '''
        Rewrite a gender-fair sentence to use generic female forms using rules.
        '''
        sent = self._rewrite_nouns_female(sent)
        sent = self._rewrite_pronouns_female(sent)
        return sent

    @staticmethod
    def _is_subject(token: 'spacy.Token') -> bool:
        '''
        Check if the current token is a subject of a verb phrase.
        '''
        if (token.dep == nsubj or token.dep == nsubjpass) and \
           (token.head.pos == VERB or token.head.pos == AUX):
            return True
        return False

    @staticmethod
    def _is_conjunct(token: 'spacy.Token', verbs: List['spacy.Token']) -> bool:
        '''
        Check if the current token is a conjunct in a verb phrase.
        '''
        if token.dep == conj and token.head in verbs:
            return True
        return False

    @staticmethod
    def _is_auxiliary(token: 'spacy.Token', verbs: List['spacy.Token']) -> bool:
        '''
        Check if the current token is a conjunct in a verb phrase.
        '''
        if token.pos == AUX and token.head in verbs:
            return True
        return False

    def _singularize_present_simple(self, lowercase_verb: str) -> str:
        '''
        Singularize present simple verb forms.
        '''
        if lowercase_verb.endswith('y'):
            return lowercase_verb[:-1] + 'ies'

        if lowercase_verb.endswith('sh'):
            return lowercase_verb + 'es'

        if lowercase_verb.endswith('ch'):
            return lowercase_verb + 'es'

        if lowercase_verb.endswith('x'):
            return lowercase_verb + 'es'

        if lowercase_verb.endswith('z'):
            return lowercase_verb + 'es'

        if lowercase_verb.endswith('s'):
            return lowercase_verb + 'es'

        return lowercase_verb + 's'

    @staticmethod
    def _capitalize(original: str, replacement: str) -> str:
        """
        Matches the capitalization type of two strings
        """
        # Check for capitalization
        if original.istitle():
            return replacement.capitalize()
        elif original.isupper():
            return replacement.upper()

        # Otherwise, return the default replacement
        return replacement

    def _find_all_verb_candidates(self, sent: 'spacy.Doc', matches: List) -> Dict:
        '''
        Find all verbs that refer to a person and their auxiliaries.
        '''
        verbs = set()

        # Find all regular verbs that refer to a person
        for tok in sent:
            if tok.text.lower() in matches and self._is_subject(tok):
                verbs.add(tok.head)

        # Find all additional verbs that are part of conjunctions
        for tok in sent:
            if self._is_conjunct(tok, verbs):
                verbs.add(tok)

        # Check if there are any auxiliary verbs
        verb_auxiliaries = defaultdict(list)
        for verb in verbs:
            verb_auxiliaries[verb] = []

        for tok in sent:
            if self._is_auxiliary(tok, verbs):
                verb_auxiliaries[tok.head].append(tok)

        return verb_auxiliaries

    def _find_singular_forms(self, verb_auxiliaries: Dict) -> Dict:
        '''
        Find the singular forms of verbs that should be replaced.
        '''
        verb_replacements = dict()

        for verb, auxiliaries in verb_auxiliaries.items():
            if 'Number=Sing' in verb.morph:
                verb_replacements[verb] = None
                continue

            if not auxiliaries:
                if 'Tense=Past' in verb.morph:
                    # Handle past tense of "to be"
                    if verb.text.lower() == 'were':
                        verb_replacements[verb] = self._capitalize(verb.text, 'was')
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
                        singular_form = self._singularize_present_simple(verb.text)
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

    def _singularize_verb_forms(self, sent: 'spacy.Doc') -> str:
        '''
        Singularize verb forms.
        '''
        verb_auxiliaries = self._find_all_verb_candidates(sent, ['they'])
        verb_replacements = self._find_singular_forms(verb_auxiliaries)

        new_toks = []
        for tok in sent:
            if tok in verb_replacements.keys() and verb_replacements[tok]:
                new_toks.append(verb_replacements[tok])
            else:
                new_toks.append(tok.text)
            if tok.whitespace_:
                new_toks.append(tok.whitespace_)

        return ''.join(new_toks)

    def reverse(self, sent: str, norm_sent: str) -> str:
        '''
        Rewrite a gender-fair sentence to use generic male and female forms using rules.
        '''
        if not self.contains_gendered_form(norm_sent):
            return sent, norm_sent, norm_sent, norm_sent

        tagged = self.spacy_tagger(norm_sent)
        sent = self._singularize_verb_forms(tagged)

        male_form = self._reverse_male(norm_sent)
        female_form = self._reverse_female(norm_sent)

        return sent, norm_sent, female_form, male_form


class EnglishNormalizer(EnglishManipulator, Normalizer):

    def __init__(self, delimiter: str):
        super().__init__()
        self.nlp = English()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 3000000

        to_ignore = [i for i, _ in TO_EXCLUDE_MF] + [i for i, _ in TO_EXCLUDE_FF]
        self.to_ignore_re = re.compile(r'\b('+'|'.join(to_ignore)+r')\b')

    def normalize(self, sent: str) -> str:
        '''
        Change different gender-fair forms in a sentence to a specified form.
        Nothing needs to be done for English.
        '''
        if self.to_ignore_re.search(sent.lower()):
            return None, False
        if self.contains_gendered_form(sent):
            return sent, True
        return sent, False
