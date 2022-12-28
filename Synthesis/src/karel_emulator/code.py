import json

from src.karel_emulator.tokens import blocktypes

"""
Class Code. Code is stored as JSON.
"""


class Code:
    def __init__(self, type_, ast_json):
        self.astJson = ast_json
        self.type = type_
        self.block_count, self.open_body = self._numType()
        self.total_count = sum(self.block_count.values())

    @classmethod
    def parse_json(cls, code_json):
        return cls(code_json['program_type'], code_json['program_json'])

    def getJson(self):
        return self.astJson

    def __str__(self):
        return json.dumps(self.astJson, sort_keys=False, indent=2)

    def _numType(self):

        block_count = {}
        for block in blocktypes:
            block_count[block] = 0

        open_body = {"status": False}
        self._numTypeBlock(self.astJson["body"], block_count, open_body)
        return block_count, open_body["status"]

    def _numTypeBlock(self, blockJson, block_count, open_body):

        for block in blockJson:
            blockType = block['type']
            block_count[blockType] += 1

            # For loops
            if blockType == 'repeat':
                body = block['body']
                if not body:
                    open_body["status"] = True
                self._numTypeBlock(body, block_count, open_body)

            # While loops
            elif blockType == 'while':
                body = block['body']
                if not body:
                    open_body["status"] = True
                self._numTypeBlock(body, block_count, open_body)

            # RepeatUntil loops
            elif blockType == 'repeatUntil':
                body = block['body']
                if not body:
                    open_body["status"] = True
                self._numTypeBlock(body, block_count, open_body)

            # If statements
            elif blockType == 'if':
                body = block['body']
                if not body:
                    open_body["status"] = True
                self._numTypeBlock(body, block_count, open_body)

            # If/else statements
            elif blockType == 'ifElse':
                ifBody = block['ifBody']
                if not ifBody:
                    open_body["status"] = True
                elseBody = block['elseBody']
                if not elseBody:
                    open_body["status"] = True
                self._numTypeBlock(ifBody, block_count, open_body)
                self._numTypeBlock(elseBody, block_count, open_body)


if __name__ == '__main__':
    with open("../../tests/karel_emulator/test_data/code_nested_while.json", "r") as c:
        code = json.load(c)

    c = Code.parse_json(code)
    print(c)
    print(c.block_count)
    print(c.total_count)
    print(c.type)
    print(c.astJson)
    print(c.open_body)
