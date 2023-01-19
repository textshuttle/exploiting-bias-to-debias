#!/usr/bin/env python3

import sys
import json

from argparse import ArgumentParser, FileType

from gfrwriter.utils import read_lm_inputs, get_normalizer

def get_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser('Normalize gender-fair forms in given data.')
    parser.add_argument('-i', '--input',
                        default=sys.stdin,
                        type=FileType('r'),
                        help='The original file with unnormalized gender-fair forms.')
    parser.add_argument('-o', '--output',
                        default=sys.stdout,
                        type=FileType('w'),
                        help='The new file (one sentence per line) with normalized gender-fair forms.')
    parser.add_argument('-f', '--input_format',
                        default='txt',
                        type=str,
                        choices=['txt', 'json'],
                        help='The file format of the input file.')
    parser.add_argument('-l', '--lang',
                        default='de',
                        type=str,
                        choices=['de', 'en', 'en-fw'],
                        help='The language that is being processed.')
    parser.add_argument('-d', '--delimiter',
                        default='*',
                        type=str,
                        help='The delimiter to use in normalized gender-fair forms.')
    parser.add_argument('--gender_fair_only',
                        action='store_true',
                        help='If set, only outputs sentences that contain a gender-fair form.')
    parser.add_argument('--not_gender_fair_only',
                        action='store_true',
                        help='If set, only outputs sentences that do not contain gender-fair forms.')
    parser.add_argument('--disable_sentence_splitting',
                        action='store_true',
                        help='If set, no sentence splitting is done.')
    return parser


def process(line: str,
            normalizer: 'Normalizer',
            disable_sentence_splitting: bool) -> str:
    '''
    Sentence split, then run processing steps.
    '''
    normalized = []
    line = line.rstrip()
    if not disable_sentence_splitting:
        sents = normalizer.nlp(line).sents
    else:
        sents = [line]
    for sent in sents:
        if not disable_sentence_splitting:
            sent = sent.text
        norm_sent, is_gender_fair = normalizer.normalize(sent)
        normalized.append((norm_sent, is_gender_fair))
    return normalized


def normalize() -> None:
    '''
    Normalize all gender-fair forms in the input.
    '''
    parser = get_parser()
    args = parser.parse_args()

    normalizer = get_normalizer(args.lang)(args.delimiter)

    lines = args.input if args.input_format == 'txt' else read_lm_inputs(args.input)

    disable_sentence_splitting = args.disable_sentence_splitting

    for line in lines:
        sents = process(line, normalizer, disable_sentence_splitting)
        for norm_sent, is_gender_fair in sents:
            # if no arg set, we print all sentences
            if not args.gender_fair_only and not args.not_gender_fair_only:
                args.output.write(f'{norm_sent}\n')

            # if gender_fair_only set, we print only sentences that were changed
            # or already contained a gender star form
            elif args.gender_fair_only and is_gender_fair:
                args.output.write(f'{norm_sent}\n')

            # if not_gender_fair_only set, we print only unchanged sentences
            # that do not contain a gender star form
            elif args.not_gender_fair_only and not is_gender_fair:
                args.output.write(f'{norm_sent}\n')


if __name__ == '__main__':
    normalize()
