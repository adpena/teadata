from teadata.query import Query


class Dummy:
    def __init__(self, payload):
        self._payload = payload

    def to_dict(self, *, include_meta=True, include_geometry=False):
        return dict(self._payload)


class Campus:
    def __init__(self, name):
        self.name = name

    def to_dict(self, *, include_meta=True, include_geometry=False):
        return {
            "name": self.name,
            "campus_number": "001001",
            "rating": "A",
        }


def test_query_to_df_filters_columns():
    query = Query([Dummy({"alpha": 1, "beta": 2})], repo=None)

    df = query.to_df(columns=["alpha"])

    assert list(df.columns) == ["alpha"]


def test_query_to_df_filters_tuple_columns():
    campus = Campus("Test Campus")
    query = Query([(campus, None, 2.5)], repo=None)

    df = query.to_df(columns=["campus_name", "distance_miles"])

    assert list(df.columns) == ["campus_name", "distance_miles"]


def test_query_map_is_chainable():
    query = Query([1, 2, 3], repo=None)

    out = query >> ("map", lambda x: x + 1) >> ("take", 2)

    assert isinstance(out, Query)
    assert out.to_list() == [2, 3]
