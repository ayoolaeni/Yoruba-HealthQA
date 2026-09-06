from src.eval.error_taxonomy import select_lowest_scoring, suggest_flags


def test_suggest_flags_detects_diacritic_error():
    ref = "Àrùn ibà máa ń fa ibà gbígbóná àti orí fífọ́."
    hyp = "Arun iba maa n fa iba gbigbona ati ori fifo."  # diacritics fully stripped
    flags = suggest_flags(hyp, ref)
    assert flags.diacritic_error is True


def test_suggest_flags_no_diacritic_error_on_exact_match():
    text = "Àrùn ibà máa ń fa ibà gbígbóná àti orí fífọ́."
    flags = suggest_flags(text, text)
    assert flags.diacritic_error is False


def test_suggest_flags_detects_code_switching():
    ref = "Àrùn ibà máa ń fa ibà gbígbóná àti orí fífọ́ pẹ̀lú àárẹ̀ ara."
    hyp = "The doctor said this virus causes a very severe complicated fever."
    flags = suggest_flags(hyp, ref)
    assert flags.code_switching is True


def test_suggest_flags_detects_omission():
    ref = "Àrùn ibà máa ń fa ibà gbígbóná àti orí fífọ́ àti àárẹ̀ ara jíjinlẹ̀ dáadáa."
    hyp = "Ibà ni."
    flags = suggest_flags(hyp, ref)
    assert flags.omission is True


def test_suggest_flags_no_omission_on_comparable_length():
    ref = "Àrùn ibà máa ń fa ibà gbígbóná àti orí fífọ́."
    hyp = "Ibà ń fa ara gbígbóná àti orí líle."
    flags = suggest_flags(hyp, ref)
    assert flags.omission is False


def test_suggest_flags_detects_disfluency_repeated_bigram():
    ref = "Ibà ń fa orí fífọ́."
    hyp = "Ibà ń fa orí fífọ́ ìbà ń fa orí fífọ́ lẹ́ẹ̀kansi."
    flags = suggest_flags(hyp, ref)
    assert flags.disfluency is True


def test_suggest_flags_no_disfluency_on_normal_text():
    ref = "Ibà ń fa orí fífọ́."
    hyp = "Ibà ń fa orí fífọ́ àti ara gbígbóná."
    flags = suggest_flags(hyp, ref)
    assert flags.disfluency is False


def test_select_lowest_scoring_returns_worst_n_sorted():
    items = [{"id": "a", "chrf": 90.0}, {"id": "b", "chrf": 10.0}, {"id": "c", "chrf": 50.0}]
    worst = select_lowest_scoring(items, "chrf", n=2)
    assert [x["id"] for x in worst] == ["b", "c"]


def test_select_lowest_scoring_returns_all_if_fewer_than_n():
    items = [{"id": "a", "chrf": 10.0}]
    worst = select_lowest_scoring(items, "chrf", n=5)
    assert len(worst) == 1
