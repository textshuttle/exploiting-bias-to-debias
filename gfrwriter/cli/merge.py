#!/usr/bin/env python3

import sys

from argparse import ArgumentParser, FileType

from gfrwriter.utils import get_merger


def get_parser() -> ArgumentParser:
    '''
    Parse arguments via command-line.
    '''
    parser = ArgumentParser('Sync gender-fair and round-trip segments.')
    parser.add_argument('-i', '--input',
                        required=True,
                        type=FileType('r'),
                        help='The original file (one sentence per line) with gender-fair forms.')
    parser.add_argument('-r', '--round_trip',
                        required=True,
                        type=FileType('r'),
                        help='The round-trip translated file (one sentence per line).')
    parser.add_argument('-o', '--output',
                        default=sys.stdout,
                        type=FileType('w'),
                        help='The new file (one sentence per line) with merged round-trip translations.')
    parser.add_argument('-l', '--lang',
                        default='de',
                        type=str,
                        choices=['de'],
                        help='The language that is being processed.')
    parser.add_argument('-d', '--delimiter',
                        default='*',
                        type=str,
                        help='The delimiter to use in normalized gender-fair forms.')
    return parser


def merge() -> None:
    '''
    Sync original gender-fair segments with round-trip translation for more consistency.
    '''
    parser = get_parser()
    args = parser.parse_args()

    merger = get_merger(args.lang)(args.delimiter)

    for gender_fair_sent, rt_sent in zip(args.input, args.round_trip):
        merged_sent = merger.merge(gender_fair_sent.rstrip(), rt_sent.rstrip())
        args.output.write(f'{merged_sent}\n')


if __name__ == '__main__':
    reverse()
