RELATION_TYPES: tuple[str, ...] = (
    'ガ',
    'ヲ',
    'ニ',
    'ガ２',
    'デ',
    'カラ',
    'ノ',
    'ノ？',
    '修飾',
    '=',
)
RELATION_TYPES += tuple(f'{k}≒' for k in RELATION_TYPES)