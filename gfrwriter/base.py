#!/usr/bin/env python3

class BaseManipulator(object):

    def get_reverser(delimiter: str,
                     reverse_approach: str) -> 'BaseManipulator':
        return NotImplementedError

    def contains_gendered_form(self, sent: str) -> bool:
        return NotImplementedError


class RuleBasedReverser(BaseManipulator):

    def reverse(self, sent: str, gender: str) -> str:
        '''
        Rewrite a gender-fair sentence to use generic male and female forms using rules.
        '''
        return NotImplementedError


class Merger(BaseManipulator):

    def merge(self, sent: str, trans: str) -> str:
        '''
        Merge non-gender-fair form into gender-fair sentence if the form matches.
        '''
        return NotImplementedError


class Normalizer(BaseManipulator):

    def normalize(self, sent: str) -> str:
        '''
        Change different gender-fair forms in a sentence to a specified form.
        '''
        return NotImplementedError
