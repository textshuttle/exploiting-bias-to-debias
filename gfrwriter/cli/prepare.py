#!/usr/bin/env python3

import sys
import json
import gzip
import codecs

from typing import List, TextIO, Dict
from argparse import ArgumentParser, FileType
from itertools import repeat
from collections import Counter

from gfrwriter.utils import read_lm_inputs, read_gz_inputs, read_oscar_inputs, get_normalizer, get_manipulator

SEEN = set()

def get_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser('Process monolingual data and prepare a parallel corpus of non-gender-fair / gender-fair data to gender-fair data.')
    parser.add_argument('-i', '--input',
                        required=True,
                        type=str,
                        help='The original monolingual data.')
    parser.add_argument('-o', '--output',
                        default='train',
                        type=str,
                        help='The file prefix for the parallel output data.')
    parser.add_argument('-d', '--data_dict',
                        default='data/german_seeds.json',
                        type=FileType('r'),
                        help='The seed dictionary used for creating data.')
    parser.add_argument('-f', '--input_format',
                        default='txt',
                        type=str,
                        choices=['txt', 'json', 'gz', 'jsonl'],
                        help='The file format of the input file.')
    parser.add_argument('-l', '--lang',
                        default='de',
                        type=str,
                        choices=['de', 'en', 'en-fw'],
                        help='The language that is being processed.')
    parser.add_argument('-a', '--reverse_approach',
                        default='rule-based',
                        type=str,
                        choices=['rule-based', 'round-trip'],
                        help='How to generate the generic male or female forms.')
    return parser


def create_parallel_data(line: str,
                         animated_nouns: Dict[str,int],
                         normalizer: 'Normalizer',
                         reverser: 'Reverser') -> List[str]:
    '''
    Sentence split, then run processing steps and write parallel data.
    '''
    srcs = []
    trgs = []
    labels = []
    line = line.rstrip()

    # Split segments into sentences
    for sent in normalizer.nlp(line).sents:
        sent = sent.text

        # Only process unique sentences
        sent_hash = hash(sent)
        if sent_hash in SEEN:
            continue
        SEEN.add(sent_hash)

        # Normalize to gender-fair star form
        norm_sent, is_gender_fair = normalizer.normalize(sent)

        # Cases where an ignore rule is triggered in normalize
        if not norm_sent:
            continue

        # Revert to generic male form and generic female form
        if is_gender_fair:

            if reverser.type == 'rule-based':
                sent, norm_sent, female_form, male_form = reverser.reverse(sent, norm_sent)

                srcs.append(male_form)
                trgs.append(norm_sent)
                labels.append('gendered-male')

                srcs.append(female_form)
                trgs.append(norm_sent)
                labels.append('gendered-female')

            srcs.append(sent)
            trgs.append(norm_sent)
            labels.append('copy-gf')

        # If not gender-fair apply additional check if segment contains animated_nouns
        else:
            if all(noun.lower() not in sent.lower() for noun in animated_nouns) and \
               all(f' {pronoun} ' not in sent and not sent.startswith(pronoun[0].upper()+pronoun[1:]) for pronoun, _ in reverser.pronouns_to_exclude):
                srcs.append(sent)
                trgs.append(norm_sent)
                labels.append('copy-no-gf')

    return srcs, trgs, labels


def prepare() -> None:
    '''
    Process monolingual data and create a non-gender-fair / gender-fair
    parallel corpus.
    '''
    parser = get_parser()
    args = parser.parse_args()

    # Load animated nouns for identifying non-animated segments
    animated_nouns = json.load(args.data_dict)

    delimiter = '@@GFM@@' if args.reverse_approach == 'rule-based' else '*'
    normalizer = get_normalizer(args.lang)(delimiter)
    reverser = get_manipulator(args.lang).get_reverser(delimiter,
                                                       args.reverse_approach)

    if args.input_format == 'txt':
        open_file = codecs.open(args.input, 'r', encoding='utf-8',
                 errors='replace')
        lines = open_file
    elif args.input_format == 'json':
        open_file = open(args.input, 'r')
        lines = read_lm_inputs(open_file)
    elif args.input_format == 'gz':
        open_file = gzip.open(args.input)
        lines = read_gz_inputs(open_file)
    elif args.input_format == 'jsonl':
        open_file = open(args.input, 'r')
        lines = read_oscar_inputs(open_file)

    with open(f'{args.output}.gf.src', 'w') as src_gf, \
         open(f'{args.output}.gf.trg', 'w') as trg_gf, \
         open(f'{args.output}.ngf.src', 'w') as src_ngf, \
         open(f'{args.output}.ngf.trg', 'w') as trg_ngf:

        for line in lines:
            srcs, trgs, labels = create_parallel_data(line,
                                                      animated_nouns,
                                                      normalizer,
                                                      reverser)
            for src, trg, label in zip(srcs, trgs, labels):
                if label == 'copy-no-gf':
                    src_ngf.write(f'{src}\n')
                    trg_ngf.write(f'{trg}\n')
                else:
                    src_gf.write(f'{src}\n')
                    trg_gf.write(f'{trg}\n')

    lines.close()
    open_file.close()

if __name__ == '__main__':
    prepare()
