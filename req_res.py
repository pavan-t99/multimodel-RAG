class Request:
    def __init__(self, query: str):
        self.query = query

    def __str__(self):
        return f"Request(query={self.query})"

    def __repr__(self):
        return str(self)


class Response:
    def __init__(self, request: Request):
        self.request = request
        self.sources = None
        self.summary = None

    def __str__(self):
        return f"Response(summary={self.summary})"

    def __repr__(self):
        return str(self)