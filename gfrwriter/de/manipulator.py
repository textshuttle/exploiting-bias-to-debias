#!/usr/bin/env python3

import re
from typing import List, Iterator, Tuple, Any

from difflib import get_close_matches

from spacy.lang.de import German

from gfrwriter.base import BaseManipulator, \
                           Merger, \
                           Normalizer


PERS_PRONOUNS_MF = [('er', 'sie'), ('ihn', 'sie'), ('ihm', 'ihr')]
PERS_PRONOUNS_FF = [('sie','er'), ('sie', 'ihn'), ('ihr', 'ihm')]

POSS_PRONOUNS_MF = [('mein', 'meine'), ('mein', 'e'), ('meinen', 'meine'),
                    ('meinem', 'meiner'), ('meinem', 'r'),
                    ('meines', 'meiner'), ('meines', 'r'),
                    ('dein', 'deine'),  ('dein', 'e'), ('deinen', 'deine'),
                    ('deinem', 'deiner'),  ('deinem', 'r'),
                    ('deines', 'deiner'), ('deines', 'r'),
                    ('sein', 'seine'), ('sein', 'e'),  ('seinen', 'seine'),
                    ('seinem', 'seiner'),  ('seinem', 'r'),
                    ('seines', 'seiner'),  ('seines', 'r'),
                    ('ihr', 'ihre'), ('ihr', 'e'), ('ihren', 'ihre'),
                    ('ihrem', 'ihrer'), ('ihrem', 'r'),
                    ('ihres', 'ihrer'), ('ihres', 'r'),
                    ('unser', 'unsere'), ('unser', 'e'), ('unseren', 'unsere'),
                    ('unserem', 'unserer'), ('unserem', 'r'),
                    ('unseres', 'unserer'), ('unseres', 'r'),
                    ('euer', 'eure'), ('euren', 'eure'),
                    ('eurem', 'eurer'), ('eurem', 'r'),
                    ('eures', 'eurer'), ('eures', 'r'),
                    ('sein', 'ihr'), ('seine', 'ihre'),
                    ('seinen', 'ihren'), ('seinem', 'ihrem'),
                    ('seines', 'ihres'), ('seiner', 'ihrer')]

POSS_PRONOUNS_FF = [('meine', 'mein'), ('meine', 'meinen'), ('meine', 'n'),
                    ('meiner', 'meinem'), ('meiner', 'm'),
                    ('meiner', 'meines'), ('meiner', 's'),
                    ('deine', 'dein'), ('deine', 'deinen'), ('deine', 'n'),
                    ('deiner', 'deinem'), ('deiner', 'm'),
                    ('deiner', 'deines'), ('deiner', 's'),
                    ('seine', 'sein'), ('seine', 'seinen'), ('seine', 'n'),
                    ('seiner', 'seinem'), ('seiner', 'm'),
                    ('seiner', 'seines'), ('seiner', 's'),
                    ('ihre', 'ihr'), ('ihre', 'ihren'), ('ihre', 'n'),
                    ('ihrer', 'ihrem'), ('ihrer', 'm'),
                    ('ihrer', 'ihres'), ('ihrer', 's'),
                    ('unsere', 'unser'), ('unsere', 'unseren'), ('unsere', 'n'),
                    ('unserer', 'unserem'), ('unserer', 'm'),
                    ('unserer', 'unseres'), ('unserer', 's'),
                    ('eure', 'euer'), ('eure', 'euren'), ('eure', 'n'),
                    ('eurer', 'eurem'), ('eurer', 'm'),
                    ('eurer', 'eures'), ('eurer', 's'),
                    ('ihr', 'sein'), ('ihre', 'seine'),
                    ('ihren', 'seinen'), ('ihrem', 'seinem'),
                    ('ihres', 'seines'), ('ihrer', 'seiner')]

RELA_PRONOUNS_MF = [('der','die'), ('den', 'die'),
                    ('dem', 'der'), ('des', 'der'), ('dem', 'r'),
                    ('dessen', 'deren'), ('des', 'r'),
                    ('welcher', 'welche'), ('welchen', 'welche'),
                    ('welchem', 'welcher'),  ('welchem', 'r'),
                    ('welches', 'welcher'),  ('welches', 'r')]

RELA_PRONOUNS_FF = [('die', 'der'), ('die', 'den'),
                    ('der', 'dem'), ('der', 'des'), ('der', 'm'),
                    ('deren', 'dessen'), ('der', 's'),
                    ('welche', 'welcher'), ('welche', 'r'),
                    ('welche', 'welchen'), ('welche', 'n'),
                    ('welcher', 'welchem'), ('welcher', 'm'),
                    ('welcher', 'welches'), ('welcher', 's')]

INDF_PRONOUNS_MF = [('ein', 'eine'), ('ein', 'e'), ('einer', 'eine'),
                    ('einen', 'eine'), ('einem', 'einer'), ('einem', 'r'),
                    ('eines', 'einer'), ('eines', 'r'),
                    ('irgendein', 'irgendeine'), ('irgendein', 'e'),
                    ('irgendeiner', 'irgendeine'), ('irgendeinen', 'irgendeine'),
                    ('irgendeinem', 'irgendeiner'), ('irgendeinem', 'r'),
                    ('irgendeines', 'irgendeiner'), ('irgendeines', 'r'),
                    ('kein', 'keine'), ('kein', 'e'),
                    ('keiner', 'keine'), ('keinen', 'keine'),
                    ('keinem', 'keiner'), ('keinem', 'r'),
                    ('keines', 'keiner'), ('keines', 'r'),
                    ('jeder', 'jede'), ('jeden', 'jede'),
                    ('jedem', 'jeder'), ('jedem', 'r'),
                    ('jedes', 'jeder'), ('jedes', 'r')]

INDF_PRONOUNS_FF = [('eine', 'ein'), ('eine', 'einer'), ('eine', 'r'),
                    ('eine', 'einen'), ('eine', 'n'),
                    ('einer', 'einem'), ('einer', 'm'),
                    ('einer', 'eines'), ('einer', 's'),
                    ('irgendeine', 'irgendein'),
                    ('irgendeine', 'irgendeiner'), ('irgendeine', 'r'),
                    ('irgendeine', 'irgendeinen'), ('irgendeine', 'n'),
                    ('irgendeiner', 'irgendeinem'), ('irgendeiner', 'm'),
                    ('irgendeiner', 'irgendeines'), ('irgendeiner', 's'),
                    ('keine', 'kein'), ('keine', 'keiner'), ('keine', 'r'),
                    ('keine', 'keinen'), ('keine', 'n'),
                    ('keiner', 'keinem'), ('keiner', 'm'),
                    ('keiner', 'keines'), ('keiner', 's'),
                    ('jede', 'jeder'), ('jede', 'r'),
                    ('jede', 'jeden'), ('jede', 'n'),
                    ('jeder', 'jedem'), ('jeder', 'm'),
                    ('jeder', 'jedes'), ('jeder', 's')]

DEMO_PRONOUNS_MF = [('dieser', 'diese'), ('diesen', 'diese'),
                    ('diesem', 'dieser'), ('diesem', 'r'),
                    ('dieses', 'dieser'), ('dieses', 'r'),
                    ('jener', 'jene'), ('jenen', 'jene'),
                    ('jenem', 'jener'), ('jenem', 'r'),
                    ('jenes', 'jener'), ('jenes', 'r'),
                    ('derjenige', 'diejenige'), ('denjenigen', 'diejenige'),
                    ('demjenigen', 'derjenigen'), ('desjenigen', 'derjenigen'),
                    ('derselbe', 'dieselbe'), ('denselben', 'dieselbe'),
                    ('demselben', 'derselben'), ('desselben', 'derselben')]

DEMO_PRONOUNS_FF = [('diese', 'dieser'), ('diese', 'r'),
                    ('diese', 'diesen'), ('diese', 'n'),
                    ('dieser', 'diesem'), ('dieser', 'm'),
                    ('dieser', 'dieses'), ('dieser', 's'),
                    ('jene', 'jener'), ('jene', 'r'),
                    ('jene', 'jenen'), ('jene', 'n'),
                    ('jener', 'jenem'), ('jener', 'm'),
                    ('jener', 'jenes'), ('jener', 's'),
                    ('diejenige', 'derjenige'), ('diejenige', 'denjenigen'),
                    ('derjenigen', 'demjenigen'), ('derjenigen', 'desjenigen'),
                    ('dieselbe', 'derselbe'), ('dieselbe', 'denselben'),
                    ('derselben', 'demselben'), ('derselben', 'desselben')]

PREPOSITIONS_MF = [('vom', 'von'), ('vom', 'n'),
                   ('zum', 'zur'), ('zum', 'r')]

PREPOSITIONS_FF = [('von', 'vom'), ('von', 'm'),
                   ('zur', 'zum'), ('zur', 'm')]


class GermanManipulator(BaseManipulator):

    def __init__(self, delimiter: str = '@@GFM@@'):
        self.type = 'manipulator'

        # The token that will be used as a placeholder for other gender-fair forms
        self.delimiter = delimiter

        # Delimiters that are typically used for gender-fair forms
        # does not contain ":" because this delimiter is too noisy
        self.delimiters = ['*', '_', '/', ' / ', ' oder ', ' bzw ', ' bzw. ', self.delimiter]

        # Pronouns that are typically gendered
        self.pronoun_pairs_mf = PERS_PRONOUNS_MF + POSS_PRONOUNS_MF + RELA_PRONOUNS_MF \
                                + INDF_PRONOUNS_MF + DEMO_PRONOUNS_MF + PREPOSITIONS_MF

        self.pronoun_pairs_ff = PERS_PRONOUNS_FF + POSS_PRONOUNS_FF + RELA_PRONOUNS_FF \
                                + INDF_PRONOUNS_FF + DEMO_PRONOUNS_FF + PREPOSITIONS_FF

        self.pronouns_to_exclude = PERS_PRONOUNS_MF + PERS_PRONOUNS_FF

        # Create regexes with all delimiters
        self.pronoun_pairs_mf = self._add_upper_and_create_regex(self.pronoun_pairs_mf, self.delimiters)
        self.pronoun_pairs_ff = self._add_upper_and_create_regex(self.pronoun_pairs_ff, self.delimiters)

        self.pronoun_pairs = self.pronoun_pairs_mf + self.pronoun_pairs_ff

        # Regexes that identify gender-fair noun forms
        self.re_binneni_plural = re.compile(r'(\w+)Innen')
        self.re_binneni_singular = re.compile(r'\b((?!Linked|Check))(\w+)In\b')

        self.re_gap_singular = re.compile(r'_in\b')
        self.re_gap_plural = re.compile(r'_innen')
        self.re_slash_singular = re.compile(r' ?/ ?in\b')
        self.re_slash_plural = re.compile(r' ?/ ?innen')

        self.re_star_plural = re.compile(r'\*innen')
        self.re_star_singular = re.compile(r'\*in')
        self.re_star = re.compile(r'\*(in(nen)?)?')
        self.re_star_to_match = re.compile(r'\*')

        self.re_marker_plural = self.re_star_plural if delimiter == '*' else re.compile(f'{self.delimiter}innen')
        self.re_marker_singular = self.re_star_singular if delimiter == '*' else re.compile(f'{self.delimiter}in')
        self.re_marker = self.re_star if delimiter == '*' else re.compile(f'{self.delimiter}'+r'(in(nen)?)?')

        self.re_noun_composita = re.compile(r'innen-?\w+')
        self.re_noun_dash = re.compile(r'-/innen')

        self.re_pair_ff = re.compile(r'(\S{2,})innen und -?\1(?!innen)(en|e|n)?')
        self.re_pair_mf = re.compile(r'(\S{2,})(?!innen)(\S+)? und -?\1innen')
        self.re_pair_singular_ff = re.compile(r'(\S{2,})in (oder|bzw|bzw\.) -?\1(?!in)(en|s)?')
        self.re_pair_singular_mf = re.compile(r'(\S{2,})(?!in)(\S+)? (oder|bzw|bzw\.) -?\1in')

        # Regexes that identify gender-fair adjective forms
        self.re_adj_nom = [re.compile(f'e\\{d}r'+r'\b') for d in self.delimiters]
        self.re_adj_acc = [re.compile(f'e\\{d}n'+r'\b') for d in self.delimiters]

    @staticmethod
    def _add_all_delimiters(k: str, v: str, delimiters: List[str]) -> List[Tuple]:
        regexes = []
        for d in delimiters:
            regexes.append(re.compile(r'\b'+f'{k}\\{d}{v}'+r'\b'))
        return regexes

    def _add_upper_and_create_regex(self,
                                    pairs: List[Tuple],
                                    delimiters: List[str]) -> List[Tuple]:
        upper_pairs = []
        for k,v in pairs:
            k_upper = k.upper()
            v_upper = v.upper()
            k_cap = k[0].upper()+k[1:]
            v_cap = v[0].upper()+v[1:]

            upper_pairs.append((k, v, self._add_all_delimiters(k, v, delimiters)))
            upper_pairs.append((k_upper,
                               v_upper,
                               self._add_all_delimiters(k_upper, v_upper, delimiters)))
            upper_pairs.append((k_cap,
                               v_cap,
                               self._add_all_delimiters(k_cap, v_cap, delimiters)))
            upper_pairs.append((k_cap,
                               v,
                               self._add_all_delimiters(k_cap, v, delimiters)))

        return upper_pairs

    def get_reverser(delimiter: str,
                     reverse_approach: str) -> 'BaseManipulator':
        if reverse_approach == 'round-trip':
            return GermanManipulator(delimiter)
        else:
            raise NotImplementedError

    def _contains_gendered_noun(self, sent: str) -> bool:
        # match nouns with delimiters, e.g. "Student*innen"
        for d in self.delimiters:
            if d not in sent:
                continue
            if d != ' oder ' and f'{d}in' in sent:
                    return True

        # match Binnen-I nouns, e.g. "StudentInnen"
        if self.re_binneni_plural.search(sent):
            return True
        if self.re_binneni_singular.search(sent):
            return True
        # match noun pair forms, e.g. "Studenten und Studentinnen"
        if self.re_pair_ff.search(sent):
            return True
        if self.re_pair_mf.search(sent):
            return True
        # match noun pair forms, e.g. "Student oder Studentin"
        if self.re_pair_singular_ff.search(sent):
            return True
        if self.re_pair_singular_mf.search(sent):
            return True

        return False

    def _contains_gendered_adj(self, sent: str)-> bool:
        for r in self.re_adj_nom:
            if r.search(sent): # matches neue/n
                return True
        for r in self.re_adj_acc:
            if r.search(sent): # matches neue/r
                return True
        return False

    def _contains_gendered_pronoun(self, sent: str)-> bool:
        for k, v, regexes in self.pronoun_pairs:
            if k not in sent:
                continue
            for r in regexes:
                if r.search(sent):
                    return True
        return False

    def contains_gendered_form(self, sent: str) -> bool:
        if self._contains_gendered_pronoun(sent):
            return True
        if self._contains_gendered_noun(sent):
            return True
        if self._contains_gendered_adj(sent):
            return True
        return False


class GermanMerger(GermanManipulator, Merger):

    def __init__(self,
                 delimiter: str):
        super().__init__(delimiter)

    def merge(self, sent: str, trans: str) -> str:
        '''
        Merge non-gender-fair form into gender-fair sentence if the form matches.
        '''
        sent_toks = sent.split()
        trans_toks = trans.split()
        sent_changed = sent
        changes = []
        for tok in sent_toks:
            if '*' in tok:
                left, right, *rest = tok.split('*')
                tok = self.re_star_to_match.sub(r'\*', tok)
                # Find tokens in translation closest to token of interest
                close_matches = get_close_matches(self.re_marker.sub('', tok), trans_toks, cutoff=0.6)
                # First check if the left or right part occurs in the round-tripped sentence.
                # Otherwise, if there is at least one close match, replace the token in the gender-fair sentence
                if left in trans_toks:
                    try:
                        sent_changed = re.sub(tok.strip('()'), left.strip('()'), sent_changed)
                        changes.append(True)
                    except:
                        pass
                elif len(right) > 1 and right != 'in' and right != 'innen' and right in trans_toks:
                    try:
                        sent_changed = re.sub(tok.strip('()'), right.strip('()'), sent_changed)
                        changes.append(True)
                    except:
                        pass
                elif len(close_matches) > 0:
                    try:
                        sent_changed = re.sub(tok.strip('()'), close_matches[0].strip('()'), sent_changed)
                        changes.append(True)
                    except:
                        return trans
                        continue
                else:
                    return trans
        if sent_changed != sent or (not all(changes)):
            return sent_changed
        else:
            return sent


class GermanNormalizer(GermanManipulator, Normalizer):

    def __init__(self, delimiter: str):
        super().__init__(delimiter)
        self.nlp = German()
        self.nlp.add_pipe("sentencizer")
        self.nlp.max_length = 3000000

    def _map_pronouns_to_form(self, sent: str) -> str:
        '''
        Change pronouns with different gender-fair forms to a specified form.
        e.g. "ein/e" -> "ein*e"
        '''
        for k,v,regexes in self.pronoun_pairs:
            if k in sent:
                for r in regexes:
                    sent = r.sub(f'{k}{self.delimiter}{v}', sent)
        return sent

    def _map_nouns_to_form(self, sent: str) -> str:
        '''
        Change nouns with different gender-fair forms to a specified form.
        e.g. "Student_in" -> "Student*in"
        '''
        if 'in' in sent or 'In' in sent:
            # Plural forms
            sent = self.re_gap_plural.sub(f'{self.delimiter}innen', sent)
            sent = self.re_slash_plural.sub(f'{self.delimiter}innen', sent)
            sent = self.re_star_plural.sub(f'{self.delimiter}innen', sent)
            sent = self.re_binneni_plural.sub(r'\1'+f'{self.delimiter}innen', sent)
            sent = self.re_pair_ff.sub(r'\1'+f'{self.delimiter}innen', sent)
            sent = self.re_pair_mf.sub(r'\1'+f'{self.delimiter}innen', sent)

            # Singular forms
            sent = self.re_gap_singular.sub(f'{self.delimiter}in', sent)
            sent = self.re_slash_singular.sub(f'{self.delimiter}in', sent)
            sent = self.re_star_singular.sub(f'{self.delimiter}in', sent)
            sent = self.re_binneni_singular.sub(r'\2'+f'{self.delimiter}in', sent)
            sent = self.re_pair_singular_ff.sub(r'\1'+f'{self.delimiter}in', sent)
            sent = self.re_pair_singular_mf.sub(r'\1'+f'{self.delimiter}in', sent)

        return sent

    def _map_adj_to_form(self, sent: str) -> str:
        '''
        Change adjectives with different gender-fair forms to a specified form.
        e.g. "neue/n" -> "neue*n"
        '''
        for r in self.re_adj_nom:
            sent = r.sub(f'e{self.delimiter}r', sent)
        for r in self.re_adj_acc:
            sent = r.sub(f'e{self.delimiter}n', sent)
        return sent

    def normalize(self, sent: str) -> str:
        '''
        Change different gender-fair forms in a sentence to a specified form.
        '''
        # remove dash
        sent = self.re_noun_dash.sub('/innen', sent)

        if self.contains_gendered_form(sent):
            sent = self._map_pronouns_to_form(sent)
            sent = self._map_nouns_to_form(sent)
            sent = self._map_adj_to_form(sent)
            return sent, True
        return sent, False
