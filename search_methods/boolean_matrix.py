from search_methods.base_method import BaseSearcher

NOT_OP = "NOT"
AND_OP = "AND"
AND_NOT_OP = AND_OP + " " + NOT_OP
OR_OP = "OR"
PAREN_START = "("
PAREN_END = ")"

OPERATORS = [NOT_OP, AND_NOT_OP, AND_OP, OR_OP]
PRECEDENCE = dict(zip(OPERATORS, range(len(OPERATORS))))


class BooleanSearcher(BaseSearcher):
    def search(self):
        pass

    def _parse_query(self, query):
        if PAREN_START not in query:
            return query.split()

        paren_query = query[query.find(PAREN_START): query.find(PAREN_END) + 1]
        result_front = query[: query.find(PAREN_START)].split()
        result_back = query[query.find(PAREN_END) + 1:].split()
        return result_front + [paren_query] + result_back

    def process_query(self, query):

        # Convert the infix query to a postfix notation using the Shunting Yard Algorithm
        postfix_query = self._shunting_yard(query)

        # Process the expression in the correct precedence
        # An empty posting list is inserted for the case of a blank query
        operands = [(0, list())]
        for t in postfix_query:
            if t in OPERATORS:
                # Evaluate the operation
                operand1 = operands.pop()

                if t == NOT_OP:
                    result = apply_not(operand1)
                else:
                    operand2 = operands.pop()
                    if t == AND_OP:
                        result = apply_and(operand1, operand2)
                    elif t == AND_NOT_OP:
                        # Note that the order of operands is swapped for AND NOT
                        result = apply_and_not(operand2, operand1)
                    elif t == OR_OP:
                        result = apply_or(operand1, operand2)
                operands.append(result)
            else:
                operands.append(t)

        result = operands.pop()
        return get_data(result)

    def _shunting_yard(self, query):
        tokens = self._parse_query(query)

        token_stack = list()
        result = list()

        def clear_token_stack():
            nonlocal token_stack, result
            while token_stack:
                result.append(token_stack.pop())

        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in OPERATORS:
                if token == AND_OP and tokens[i + 1] == NOT_OP:
                    token = AND_NOT_OP
                    i += 1

                if token_stack and PRECEDENCE[token_stack[-1]] > PRECEDENCE[token]:
                    clear_token_stack()
                token_stack.append(token)
            elif token.startswith(PAREN_START):
                parsed_query = self._shunting_yard(token[1: -1])
                result += parsed_query
            else:
                result.append(self.pre_processor.process(token)[0][0])

            i += 1

        clear_token_stack()

        return result
