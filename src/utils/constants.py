CASES: tuple[str, ...] = (
    'ガ',
    'ヲ',
    'ニ',
    'ガ２',
    'デ',
    'カラ',
)
CASES_ALL = CASES + tuple(f'{k}≒' for k in CASES)

RELATION_TYPES: tuple[str, ...] = CASES + (
    'ノ',
    'ノ？',
    '修飾',
    '=',
)
RELATION_TYPES_ALL = RELATION_TYPES + tuple(f'{k}≒' for k in RELATION_TYPES)
