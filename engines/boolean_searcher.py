import json
import os
import re

import numpy as np

from engines.base_searcher import BaseSearcher
from engines.utils import create_boolean_matrix, BooleanDataOut

NOT = 'not'
AND = 'and'
OR = 'or'


class BooleanSearcher(BaseSearcher):
    _MATRIX_PATH = 'matrices'
    _MATRIX_NAME = 'matrix.npz'
    _HEADER_NAME = 'header.json'

    def __init__(self, data, build: bool = True):
        super().__init__(data)
        self.output_cls = BooleanDataOut

        matrix_path = os.path.join(self._MATRIX_PATH, self._MATRIX_NAME)
        header_path = os.path.join(self._MATRIX_PATH, self._HEADER_NAME)

        if not (build or os.path.exists(matrix_path) or os.path.exists(header_path)):
            raise ValueError

        if not os.path.exists(self._MATRIX_PATH):
            os.mkdir(self._MATRIX_PATH)

        self.matrix, self.header = self._get_matrix(build, matrix_path, header_path)

    def _get_matrix(self, build, matrix_path, header_path):
        return create_boolean_matrix(
            self.data, matrix_path, header_path
        ) if build else (np.load(matrix_path)['matrix'], json.load(open(header_path, 'r')))

    @property
    def _all_words(self):
        """words as columns"""
        return self.header['columns']

    @property
    def _all_urls(self):
        """urls as rows"""
        return self.header['rows']

    def _get_column(self, word):
        try:
            index = self._all_words.index(word[1])
            matrix = self.matrix[:, index]
            if word[0] == NOT:
                matrix = ~matrix
            return matrix
        except ValueError:
            return np.zeros(len(self._all_urls), dtype=bool)

    def _operate(self, op1, op2, operator):
        if operator == AND:
            return op1 & op2

        if operator == OR:
            return op1 | op2

    def _handle_not(self, tokens):
        new_tokens = list()

        i = 0
        n = len(tokens)
        while i < n:
            token = tokens[i]
            if i + 1 < n and token in [AND, OR]:
                new_tokens.append(token)
            elif i + 1 < n and token == NOT:
                new_tokens.append((NOT, tokens[i + 1]))
                i += 1
            else:
                new_tokens.append(('', token))
            i += 1
        return new_tokens

    def process_query(self, query):
        query = re.sub('\\W+', ' ', query).strip()
        words, operators = list(), list()
        tokens = self._handle_not(query.split())

        for token in tokens:
            if isinstance(token, str):
                operators.append(token)
            else:
                words.append(token)

        return words, operators

    def search(self, query, k):
        words, operators = self.process_query(query)
        assert len(words) == len(operators) + 1
        n = len(words)
        if n == 0:
            return None
        if n < 2:
            return self._get_results(self._get_column(words[0]), k)

        op1, op2 = self._get_column(words[0]), self._get_column(words[1])
        operator = operators[0]
        result = self._operate(op1, op2, operator)

        for i, token in enumerate(words[2:]):
            op2 = self._get_column(token)
            result = self._operate(result, op2, operators[i + 1])

        return self._get_results(result, k)

    def _get_results(self, column, k):
        indexes = column.nonzero()[0]
        urls = self._all_urls
        results = [self.output_cls(url=urls[i]) for i in indexes]
        return results[:k] if k < len(results) else results


if __name__ == '__main__':
    data = [{'content': 'salam manam khubam.', 'url': '1'}, {'content': 'to chetori', 'url': '2'},
            {'content': 'boro baba khubi', 'url': '3'}]
    s = BooleanSearcher(data, build=False)
    q = 'salam and not khubi or not baba'
    print(s.search(q, 2))
