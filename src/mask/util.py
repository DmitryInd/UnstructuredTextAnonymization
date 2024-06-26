from enum import Enum
from typing import Tuple, List, Iterable, Dict

# from mask.base import MaskFn
# from mask.n_gram import RandomNgramsMaskFn
# from mask.personal_entity import PersonalEntityMaskFn
#
#
# def get_mask_function(mask_type, *args, **kwargs) -> Optional[MaskFn]:
#     if mask_type == 'n_gram':
#         return RandomNgramsMaskFn(*args, **kwargs)
#     elif mask_type == 'personal_entity':
#         return PersonalEntityMaskFn(*args, **kwargs)
#
#     return None


def align_char_mask_to_tokens(d: str, d_toks: List[str], masked_char_spans: List[Tuple[Enum, int, int]],
                              ensure_valid_bounds_in_spans=True, ensure_nonoverlapping_spans=True) \
        -> List[Tuple[Enum, int, int]]:
    """
    :param d: текст документа в формате строки
    :param d_toks: текстовые токены, на которые разбит документ
    :param masked_char_spans: маски в формате (тип, сдвиг по символам, длина по символам)
    :param ensure_valid_bounds_in_spans: проверять ли границы масок
    :param ensure_nonoverlapping_spans: проверять ли пересечения границ масок
    :return: маски в формате (тип, сдвиг по токенам, длина по токенам)
    """
    # Find token offsets in characters
    # try:
    #     d_toks_offs = tokens_offsets(d, d_toks)
    #     assert None not in d_toks_offs
    # except:
    #     raise ValueError('Tokens could not be aligned to document')

    # Align character spans to model tokens
    x_tok_offsets, x_tok_residuals = tokens_offsets_and_residuals_memorized(d, tuple(d_toks))
    masked_token_spans = [align_charspan_to_tokenspan(d, d_toks, x_tok_offsets, x_tok_residuals, char_off, char_len)[2:]
                          for t, char_off, char_len in masked_char_spans]

    if ensure_valid_bounds_in_spans and not masked_spans_bounds_valid(masked_token_spans, len(d_toks)):
        raise ValueError('Alignment produced invalid token spans')
    if ensure_nonoverlapping_spans and masked_spans_overlap(masked_token_spans):
        raise ValueError('Alignment produced overlapping token spans')

    result = []
    for (char_t, char_off, char_len), (tok_off, tok_len) in zip(masked_char_spans, masked_token_spans):
        # Token span must contain strictly more text than the original span
        orig_span = d[char_off:char_off + char_len]
        tok_span = ''.join(d_toks[tok_off:tok_off + tok_len])
        if orig_span not in tok_span:
            # TODO Доделать обработку ошибок, случай с апострофом всё ещё обрабатывается неправильно
            print(f'Character span "{orig_span}" doesn\'t match with tokens span "{tok_span}"')

        result.append((char_t, tok_off, tok_len))

    return result


def masked_spans_bounds_valid(masked_spans, d_len):
    # masked_spans = [mask_type, mask_offset, mask_len]
    for span_off, span_len in [s[-2:] for s in masked_spans]:
        if span_off < 0 or span_len <= 0 or span_off + span_len > d_len:
            return False
    return True


def masked_spans_overlap(masked_spans):
    last_off = None
    last_len = None
    overlap = False
    # masked_spans = [mask_type, mask_offset, mask_len]
    for span_off, span_len in [s[-2:] for s in sorted(masked_spans, key=lambda x: x[-2])]:
        if last_off is not None:
            if span_off < last_off + last_len:
                overlap = True
                break
        last_off = span_off
        last_len = span_len
    return overlap


def align_charspan_to_tokenspan(x: str, x_tok: List[str], x_tok_offsets: List[int], x_tok_residuals: List[str],
                                char_offset: int, char_len: int) -> Tuple[int, int, int, int]:
    """
    Переводит запись текстового отрезка от значений в символах к значениям в токенах. Выравнивает сдвиги между собой.
    :param x: строка текста
    :param x_tok: токены, на которые разбита строка
    :param x_tok_offsets: список сдвигов на начало токенов
    :param x_tok_residuals: список оставшихся перед/между/после токенами отрезков, которые не были разбиты на токены
    :param char_offset: сдвиг отрезка по символам
    :param char_len: длина отрезка по символам
    :return: (сдвиг по символам, длина по символам, сдвиг по токенам, длина по токенам)
    """
    if len(x_tok) == 0:
        raise ValueError()
    if char_offset < 0 or char_len < 0 or (char_offset + char_len) > len(x):
        raise ValueError()

    # Build char_idx_to_token of appropriate token for each cursor index
    # NOTE: This is one greater than len(x) because cursor can be at beginning or end.
    char_idx_to_token = [0] * len(x_tok_residuals[0])
    for i in range(len(x_tok)):
        if x_tok_offsets[i] is not None:
            char_idx_to_token += [i] * (len(x_tok[i]) + len(x_tok_residuals[i + 1]))
    char_idx_to_token += [len(x_tok) - 1]

    if char_len == 0:
        token_offset = char_idx_to_token[char_offset]
        token_len = 0
        char_offset = x_tok_offsets[token_offset]
        char_len = 0
    else:
        selected_x_tok = set(char_idx_to_token[char_offset:char_offset + char_len])
        token_offset = min(selected_x_tok)
        token_end = max(selected_x_tok)
        token_len = token_end - token_offset + 1

        char_offset = x_tok_offsets[token_offset]
        char_end = x_tok_offsets[token_end] + len(x_tok[token_end])
        char_len = char_end - char_offset

    return char_offset, char_len, token_offset, token_len


def apply_masked_spans(doc: List[int], masked_spans: List[Tuple[Enum, int, int]],
                       mask_type_to_substitution: Dict[Enum, int]) -> Tuple[list, List[Tuple[Enum, list]]]:
    """
    Заменяет токены маскируемых отрезков на маскировочные токены
    :param doc: текст в формате списка id токенов
    :param masked_spans: список замаскированных отрезков текста в формате (тип, сдвиг, длина);
                         отрезки должны быть отсортированы по сдвигу в порядке возрастания!!!
    :param mask_type_to_substitution: словарь id маскировочных токенов для каждого типа маски
    :return: (контекст с убранным замаскированным текстом, список замаскированных отрезков: (тип, список токенов))
    """
    if None in doc:
        raise ValueError()

    context = doc[:]
    answers = []
    for i, (span_type, span_off, span_len) in enumerate(masked_spans):
        if span_len == 0:
            continue

        if span_off >= len(context):
            raise ValueError()

        while context[span_off] is None:
            masked_spans[i] = (span_type, span_off + 1, span_len - 1)
            _, span_off, span_len = masked_spans[i]
        answers.append((span_type, context[span_off:span_off + span_len]))
        context[span_off:span_off + span_len] = [None] * span_len

    for (_, span) in answers:
        if None in span:
            raise ValueError('Overlapping mask detected')

    for span_type, _, span_len in masked_spans:
        span_off = context.index(None)
        assert all([i is None for i in context[span_off:span_off + span_len]])
        del context[span_off:span_off + span_len]
        substitution = mask_type_to_substitution[span_type]
        context.insert(span_off, substitution)
    assert None not in context

    return context, answers


def tokens_offsets(x: str, x_tok: Iterable[str]) -> List[int]:
    """
    Ищет позиции токенов, на которые делится строка.
    :param x: строка текста
    :param x_tok: текстовые токены, на которые делится строка
    :return:
    """
    if not isinstance(x_tok, tuple):
        x_tok = tuple(x_tok)
    return tokens_offsets_and_residuals_memorized(x, x_tok)[0]


def tokens_offsets_and_residuals_memorized(x: str, x_tok: Tuple[str, ...]) -> Tuple[List[int], List[str]]:
    """
    Исследует, как текст был разбит на токены.
    :param x: строка текста
    :param x_tok: текстовые токены, на которые делится строка
    :return: (список сдвигов на начало токенов;
              список оставшихся перед/между/после токенами отрезков, которые не были разбиты на токены)
    """
    x_remaining_off = 0
    x_remaining = x[:]

    offsets = []
    residuals = []

    for i, t in enumerate(x_tok):
        try:
            t_off_in_x_remaining = x_remaining.index(t)
            t_res = x_remaining[:t_off_in_x_remaining]
            t_off = x_remaining_off + t_off_in_x_remaining
        except:
            t_off_in_x_remaining = None
            t_off = None
            t_res = ''

        offsets.append(t_off)
        residuals.append(t_res)

        if t_off is not None:
            trim = t_off_in_x_remaining + len(t)
            x_remaining_off += trim
            x_remaining = x_remaining[trim:]

    residuals.append(x_remaining)

    return offsets, residuals


def convert_masks_to_segments(doc: str, masks: List[Tuple[Enum, int, int]], other_label: str = 'O') \
        -> Tuple[List[str], List[str]]:
    """
    Конвертирует документ и набор масок для него в список отрезков и список типов отрезков
    :param doc: документ в формате строки
    :param masks: набор масок для документа: [(тип, сдвиг, длина), ...]
    :param other_label: метка для обычного текста в формате строки
    return: категории сущностей в формате [список категорий слов в документе, ...];
            исходный текст в формате [список слов в документе, ...]
    """
    category_list = []
    source_text_list = []
    cursor = 0
    for m_type, offset, length in masks:
        if offset > cursor:
            category_list.append(other_label)
            source_text_list.append(doc[cursor: offset])
        category_list.append(m_type.name)
        source_text_list.append(doc[offset: offset + length])
        cursor = offset + length
    if cursor < len(doc):
        category_list.append(other_label)
        source_text_list.append(doc[cursor:])

    return category_list, source_text_list


def convert_masked_docs_to_segments_set(masked_docs: List[Tuple[str, str, List[List[Tuple[Enum, int, int]]]]],
                                        other_label: str = 'O') -> Tuple[List[str], List[List[str]], List[List[str]]]:
    """
    Конвертирует набор замаскированных документов и набор масок для него в списки отрезков и списки типов отрезков
    :param masked_docs: [(индекс документа, текст документа,
                          список наборов масок для него: [[(тип, сдвиг, длина), ...], ...]), ...]
    :param other_label: метка для обычного текста в формате строки
    return: категории сущностей в формате [список категорий отрезков в документе, ...];
            исходный текст в формате [список отрезков в документе, ...]
    """
    record_id_list = []
    category_lists = []
    source_text_lists = []
    for record_id, doc, masks_set in masked_docs:
        for masks in masks_set:
            category_list, source_text_list = convert_masks_to_segments(doc, masks, other_label)
            record_id_list.append(record_id)
            category_lists.append(category_list)
            source_text_lists.append(source_text_list)

    return record_id_list, category_lists, source_text_lists
