import sys
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from abc import ABC
from typing import Optional


class Token(ABC):
    def __str__(self) -> str:
        raise NotImplementedError


@dataclass
class ProlongedToken(Token):
    def __str__(self):
        return ''


@dataclass
class UnclearToken(Token):
    def __str__(self):
        return ''


@dataclass
class PauseToken(Token):
    def __str__(self):
        return ''


@dataclass
class FillerToken(Token):
    text: str

    def __str__(self):
        return self.text


@dataclass
class ReactiveToken(Token):
    text: str

    def __str__(self):
        return self.text


@dataclass
class LaughToken(Token):
    text: str

    def __str__(self):
        return self.text


@dataclass
class TextToken(Token):
    text: str

    def __str__(self):
        return self.text


@dataclass
class AlphabetToken(Token):
    alphabet: str
    reading: str

    def __str__(self):
        return self.alphabet


@dataclass
class SlipToken(Token):
    correct: list[Token]
    reading: str
    original: list[Token]

    def __str__(self):
        return ''.join(str(t) for t in self.correct)


class Speaker(Enum):
    MASTER = '主人'
    ROBOT = 'ロボット'


@dataclass
class Transcription:
    """
    A class to represent a transcription
    """
    utterance: list[Token]
    speaker: Speaker
    start_time: int  # ms
    end_time: int  # ms
    duration: int  # ms


class Parser:
    def __init__(self, raw_string: str):
        self.string = raw_string

    def parse(self) -> list[Token]:
        tokens = self._tokens()
        if self.string:
            raise ValueError(f'Unexpected token: {self.string[:10]}')
        return tokens

    def _tokens(self):
        tokens = []
        while token := self._token():
            tokens.append(token)
        return tokens

    def _token(self) -> Optional[Token]:
        if self._consume('('):
            if self._consume('F'):
                self._consume(' ')
                token = FillerToken(self._consume_string())
            elif self._consume('R'):
                self._consume(' ')
                token = ReactiveToken(self._consume_string())
            elif self._consume('L'):
                self._consume(' ')
                token = LaughToken(self._consume_string())
            elif self._consume('P'):
                token = PauseToken()
            elif self._consume('?'):
                token = UnclearToken()
            else:
                raise ValueError(f'Unknown token: {self.string[:10]}')
            self._expect(')')
        elif self._consume('<'):
            if self._consume('H'):
                token = ProlongedToken()
            else:
                alphabet = self._consume_string()
                self._expect('|')
                reading = self._consume_string()
                token = AlphabetToken(alphabet, reading)
            self._expect('>')
        elif self._consume('{'):
            correct = self._tokens()
            self._expect('|')
            reading = self._consume_string()
            self._expect('|')
            original = self._tokens()
            self._expect('}')
            token = SlipToken(correct, reading, original)
        else:
            if text := self._consume_string():
                token = TextToken(text)
            else:
                return None
        return token

    def _consume(self, pattern: str) -> bool:
        if self.string.startswith(pattern):
            self.string = self.string[len(pattern):]
            return True
        return False

    def _consume_string(self) -> str:
        match = re.match(r'[^<>(){}A-Z|]*', self.string)
        self.string = self.string[match.end():]
        return match.group(0)

    def _expect(self, pattern: str):
        if self._consume(pattern) is None:
            raise ValueError(f'Expected {pattern}')


def parse_transcription_file(path: Path) -> list[Transcription]:
    transcriptions = []
    for line in path.read_text().splitlines():
        # columns:
        # １．話者
        # \t
        # ２．開始時間（時:分:秒.ミリ秒）
        # ３．開始時間（秒.ミリ秒）
        # ４．開始時間（ミリ秒）
        # ５．終了時間（時:分:秒.ミリ秒）
        # ６．終了時間（秒.ミリ秒）
        # ７．終了時間（ミリ秒）
        # ８．間隔（時:分:秒.ミリ秒）
        # ９．間隔（秒.ミリ秒）
        # １０．間隔（ミリ秒）
        # １１．発話書き起こし
        speaker, _, _, _, start_time, _, _, end_time, _, _, duration, raw_utterance = line.split('\t')
        assert int(end_time) - int(start_time) == int(duration)
        parser = Parser(raw_utterance)
        transcriptions.append(
            Transcription(parser.parse(), Speaker(speaker), int(start_time), int(end_time), int(duration))
        )
    return transcriptions


def main():
    data_dir = Path(sys.argv[1])
    transcriptions = []
    for transcription_dir in (d for d in data_dir.glob('*') if d.is_dir() and re.match(r'\d{8}-\d{8}-\d', d.name)):
        scenario_id = transcription_dir.name
        latest_file = sorted(transcription_dir.glob(f'{scenario_id}-*.txt'), key=lambda p: p.name)[-1]
        transcriptions.extend(parse_transcription_file(latest_file))
    print(transcriptions)


if __name__ == '__main__':
    main()
