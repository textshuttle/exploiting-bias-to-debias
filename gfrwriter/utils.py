#!/usr/bin/env python3

import base64
import json

from typing import TextIO, Generator, Union

from gfrwriter.de.manipulator import GermanNormalizer, GermanManipulator, GermanMerger
from gfrwriter.en.manipulator import EnglishNormalizer, EnglishManipulator
from gfrwriter.enfw.manipulator import EnglishFWNormalizer, EnglishFWManipulator

def read_lm_inputs(open_file: TextIO) -> Generator[str, None, None]:
    '''
    Read LM data from json output.
    '''
    prompts = json.load(open_file)
    for prompt in prompts:
        for generation in prompt:
            lines = generation['generated_text'].rstrip().split('\n')
            for line in lines:
                yield line


def read_gz_inputs(open_file: TextIO) -> Generator[str, None, None]:
    '''
    Read one-sent-per-line data from gz file.
    '''
    for encoded in open_file:
        line = base64.standard_b64decode(encoded).decode('utf-8')
        yield line


def read_oscar_inputs(open_file: TextIO) -> Generator[str, None, None]:
    '''
    Read OSCAR data from jsonl file.
    '''
    for f in open_file:
        f = json.loads(f)
        lines = f['content']
        for line in lines.rstrip().split('\n'):
            yield line


def get_normalizer(lang: str) -> 'Normalizer':
    '''
    Get the correct normalizer by language.
    '''
    if lang == 'de':
        return GermanNormalizer
    elif lang == 'en':
        return EnglishNormalizer
    elif lang == 'en-fw':
        return EnglishFWNormalizer
    else:
        raise NotImplementedError


def get_manipulator(lang: str) -> 'Manipulator':
    '''
    Get the correct manipulator by language.
    '''
    if lang == 'de':
        return GermanManipulator
    elif lang == 'en':
        return EnglishManipulator
    elif lang == 'en-fw':
        return EnglishFWManipulator
    else:
        raise NotImplementedError


def get_merger(lang: str) -> 'Merger':
    '''
    Get the correct merger by language.
    '''
    if lang == 'de':
        return GermanMerger
    else:
        raise NotImplementedError
