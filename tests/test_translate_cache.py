from src.model.translate import TranslationCache, content_hash, translate_with_cache


class FakeTranslator:
    """Stands in for NllbTranslator in tests -- no torch/transformers needed."""

    def __init__(self):
        self.calls: list[list[str]] = []

    def translate_batch(self, texts, max_new_tokens=256, batch_size=16):
        self.calls.append(list(texts))
        return [f"YO[{t}]" for t in texts]


def test_content_hash_is_deterministic():
    assert content_hash("hello") == content_hash("hello")
    assert content_hash("hello") != content_hash("world")


def test_cache_round_trip(tmp_path):
    cache_path = tmp_path / "cache.json"
    cache = TranslationCache(cache_path)
    assert cache.get("hello") is None
    cache.put("hello", "pele")
    assert cache.get("hello") == "pele"
    cache.save()

    reloaded = TranslationCache(cache_path)
    assert reloaded.get("hello") == "pele"


def test_translate_with_cache_only_translates_uncached_strings(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json")
    cache.put("already done", "ti pari tẹlẹ")
    translator = FakeTranslator()

    texts = ["already done", "new text", "another new text"]
    results = translate_with_cache(texts, translator, cache)

    assert results == ["ti pari tẹlẹ", "YO[new text]", "YO[another new text]"]
    assert translator.calls == [["new text", "another new text"]]


def test_translate_with_cache_second_call_translates_nothing(tmp_path):
    cache = TranslationCache(tmp_path / "cache.json")
    translator = FakeTranslator()
    texts = ["a", "b"]

    translate_with_cache(texts, translator, cache)
    results_second = translate_with_cache(texts, translator, cache)

    assert results_second == ["YO[a]", "YO[b]"]
    assert len(translator.calls) == 1  # second call hit the cache entirely


def test_translate_with_cache_handles_empty_strings():
    cache = TranslationCache.__new__(TranslationCache)
    cache._data = {}
    translator = FakeTranslator()

    results = translate_with_cache(["", "hello"], translator, cache)
    assert results == ["", "YO[hello]"]
