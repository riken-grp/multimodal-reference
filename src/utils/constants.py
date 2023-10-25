CASES: list[str] = "ガ ヲ ニ ト デ カラ ヨリ ヘ マデ ガ２ ヲ２ ニ２ ト２ デ２ カラ２ ヨリ２ ヘ２ マデ２".split()
CASES_ALL: list[str] = CASES + [f"{k}≒" for k in CASES]

RELATION_TYPES: list[str] = CASES + "ノ ノ？ 修飾 トイウ =".split()
RELATION_TYPES_ALL: list[str] = RELATION_TYPES + [f"{k}≒" for k in RELATION_TYPES]
