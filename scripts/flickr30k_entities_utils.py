# copied from https://github.com/BryanPlummer/flickr30k_entities/blob/master/flickr30k_entities_utils.py

import xml.etree.ElementTree as ET
from pathlib import Path
import re

from rhoknp import Jumanpp


def get_sentence_data(fn):
    """
    Parses a sentence file from the Flickr30K Entities dataset

    input:
      fn - full file path to the sentence file to parse
    
    output:
      a list of dictionaries for each sentence with the following fields:
          sentence - the original sentence
          phrases - a list of dictionaries for each phrase with the
                    following fields:
                      phrase - the text of the annotated phrase
                      first_word_index - the position of the first word of
                                         the phrase in the sentence
                      phrase_id - an identifier for this phrase
                      phrase_type - a list of the coarse categories this 
                                    phrase belongs to

    """
    with open(fn, 'r') as f:
        sentences = f.read().split('\n')

    annotations = []
    for sentence in sentences:
        if not sentence:
            continue

        first_word = []
        phrases = []
        phrase_id = []
        phrase_type = []
        words = []
        current_phrase = []
        add_to_phrase = False
        for token in sentence.split():
            if add_to_phrase:
                if token[-1] == ']':
                    add_to_phrase = False
                    token = token[:-1]
                    current_phrase.append(token)
                    phrases.append(' '.join(current_phrase))
                    current_phrase = []
                else:
                    current_phrase.append(token)

                words.append(token)
            else:
                if token[0] == '[':
                    add_to_phrase = True
                    first_word.append(len(words))
                    parts = token.split('/')
                    phrase_id.append(parts[1][3:])
                    phrase_type.append(parts[2:])
                else:
                    words.append(token)

        sentence_data = {'sentence': ' '.join(words), 'phrases': []}
        for index, phrase, p_id, p_type in zip(first_word, phrases, phrase_id, phrase_type):
            sentence_data['phrases'].append(
                {
                    'first_word_index': index,
                    'phrase': phrase,
                    'phrase_id': p_id,
                    'phrase_type': p_type
                }
            )

        annotations.append(sentence_data)

    return annotations


def get_sentence_data_ja(fn):
    # exapmle: 5:[/EN#550/clothing 赤い服]を着た4:[/EN#549/people 男]が6:[/EN#551/other 綱]を握って見守っている間に、1:[/EN#547/people 数人のクライマー]が2:[/EN#554/other 列]をなして3:[/EN#548/other 岩]をよじ登っている。
    tag_pat = re.compile(r'\d+:\[/EN#(?P<id>\d+)(/(?P<type>[a-z]+))+ (?P<words>[^]]+)]')
    annotations = []
    for line in Path(fn).read_text().splitlines():
        chunks = []
        raw_sentence = ''
        sidx = 0
        matches: list[re.Match] = list(re.finditer(tag_pat, line))
        for match in matches:
            # chunk 前を追加
            text = line[sidx:match.start()]
            raw_sentence += text
            chunks.append(text)
            # match の中身を追加
            raw_sentence += match.group('words')
            chunks.append({
                'phrase': match.group('words'),
                'phrase_id': match.group('id'),
                'phrase_type': match.group('type'),
            })
            sidx = match.end()
        raw_sentence += line[sidx:]
        is_word_ends = _get_is_word_ends(raw_sentence)
        sentence = ''
        phrases = []
        char_idx = word_idx = 0
        for chunk in chunks:
            if isinstance(chunk, str):
                for char in chunk:
                    sentence += char
                    if is_word_ends[char_idx]:
                        sentence += ' '
                        word_idx += 1
                    char_idx += 1
            else:
                new_phrase = ''
                chunk['first_word_index'] = word_idx
                for char in chunk['phrase']:
                    sentence += char
                    new_phrase += char
                    if is_word_ends[char_idx]:
                        sentence += ' '
                        new_phrase += ' '
                        word_idx += 1
                    char_idx += 1
                chunk['phrase'] = new_phrase.strip()
                phrases.append(chunk)
        assert 'EN' not in sentence
        annotations.append({
            'sentence': sentence,
            'phrases': phrases
        })
    return annotations


def _get_is_word_ends(sentence: str) -> list[bool]:
    morphemes = Jumanpp().apply_to_sentence(sentence).morphemes
    is_word_ends = [False] * len(sentence)
    cum_lens = -1
    for m in morphemes:
        cum_lens += len(m.text)
        is_word_ends[cum_lens] = True
    return is_word_ends


def get_annotations(fn):
    """
    Parses the xml files in the Flickr30K Entities dataset

    input:
      fn - full file path to the annotations file to parse

    output:
      dictionary with the following fields:
          scene - list of identifiers which were annotated as
                  pertaining to the whole scene
          nobox - list of identifiers which were annotated as
                  not being visible in the image
          boxes - a dictionary where the fields are identifiers
                  and the values are its list of boxes in the 
                  [xmin ymin xmax ymax] format
    """
    tree = ET.parse(fn)
    root = tree.getroot()
    size_container = root.findall('size')[0]
    anno_info = {'boxes': {}, 'scene': [], 'nobox': []}
    for size_element in size_container:
        anno_info[size_element.tag] = int(size_element.text)

    for object_container in root.findall('object'):
        for names in object_container.findall('name'):
            box_id = names.text
            box_container = object_container.findall('bndbox')
            if len(box_container) > 0:
                if box_id not in anno_info['boxes']:
                    anno_info['boxes'][box_id] = []
                xmin = int(box_container[0].findall('xmin')[0].text) - 1
                ymin = int(box_container[0].findall('ymin')[0].text) - 1
                xmax = int(box_container[0].findall('xmax')[0].text) - 1
                ymax = int(box_container[0].findall('ymax')[0].text) - 1
                anno_info['boxes'][box_id].append([xmin, ymin, xmax, ymax])
            else:
                nobndbox = int(object_container.findall('nobndbox')[0].text)
                if nobndbox > 0:
                    anno_info['nobox'].append(box_id)

                scene = int(object_container.findall('scene')[0].text)
                if scene > 0:
                    anno_info['scene'].append(box_id)

    return anno_info
