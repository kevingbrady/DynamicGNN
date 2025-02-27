

class NodeList(dict):

    def __getitem__(self, item):

        try:
            return super().__getitem__(item)

        except KeyError:
            return None

