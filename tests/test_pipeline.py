"""Tests for the schema check, attribution screen, sampling, retrieval and data adapter."""

from __future__ import annotations

import json

import pytest

from saferag.checks.attribution import CANDIDATE, CONTROL, jaccard, screen
from saferag.checks.faithfulness import RuleDecomposer, StubNLIScorer, check_faithfulness
from saferag.data.obliqa import ObliQAFormatError, adapt_record, build_passage_corpus
from saferag.generation.generator import StubGenerator, render_prompt
from saferag.generation.schema import check_schema, extract_json_block
from saferag.pilot.sample import double_annotation_subset, stratified_sample
from saferag.retrieval.hybrid import BM25Retriever, Hit, reciprocal_rank_fusion, tokenize

# --------------------------------------------------------------------------
# S1 -- schema
# --------------------------------------------------------------------------


def test_schema_accepts_valid():
    out = json.dumps({"answer": "Six years.", "cited_passage_ids": ["p1"], "obligations": []})
    assert check_schema(out).valid


def test_schema_rejects_empty_citations():
    res = check_schema(json.dumps({"answer": "x", "cited_passage_ids": []}))
    assert not res.valid
    assert "cited_passage_ids" in (res.error or "")


def test_schema_rejects_whitespace_only_citations():
    assert not check_schema(json.dumps({"answer": "x", "cited_passage_ids": ["  "]})).valid


def test_schema_rejects_malformed_json():
    res = check_schema("{not json")
    assert not res.valid
    assert res.error.startswith("json_decode_error")


def test_schema_rejects_missing_answer():
    assert not check_schema(json.dumps({"cited_passage_ids": ["p1"]})).valid


def test_schema_rejects_empty_answer():
    assert not check_schema(json.dumps({"answer": "", "cited_passage_ids": ["p1"]})).valid


def test_schema_rejects_extra_fields():
    """extra='forbid' -- an unexpected key is a schema violation, by design."""
    out = json.dumps({"answer": "x", "cited_passage_ids": ["p1"], "surprise": 1})
    assert not check_schema(out).valid


def test_schema_rejects_top_level_array():
    assert check_schema("[1, 2, 3]").error == "top_level_not_object"


def test_markdown_fence_is_stripped():
    fenced = '```json\n{"answer": "x", "cited_passage_ids": ["p1"]}\n```'
    assert extract_json_block(fenced).startswith("{")
    assert check_schema(fenced).valid


def test_schema_strips_citation_whitespace():
    res = check_schema(json.dumps({"answer": "x", "cited_passage_ids": [" p1 ", "p2"]}))
    assert res.parsed["cited_passage_ids"] == ["p1", "p2"]


# --------------------------------------------------------------------------
# S3 -- attribution screen
# --------------------------------------------------------------------------


def test_jaccard_basic():
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    assert jaccard(["a"], ["a"]) == 1.0
    assert jaccard(["a"], ["b"]) == 0.0


def test_jaccard_empty_both_is_one():
    assert jaccard([], []) == 1.0


def test_screen_no_overlap_is_candidate():
    s = screen(["p9"], ["p1", "p2"])
    assert s.pool == CANDIDATE
    assert s.n_overlap == 0


def test_screen_any_overlap_is_control():
    s = screen(["p1", "p9"], ["p1", "p2"])
    assert s.pool == CONTROL
    assert s.n_overlap == 1


def test_screen_ignores_empty_ids():
    assert screen(["", "p1"], ["p1"]).n_cited == 1


def test_screen_with_no_gold_is_candidate():
    """No gold passages -> jaccard 0 -> candidate pool."""
    assert screen(["p1"], []).pool == CANDIDATE


# --------------------------------------------------------------------------
# S2 -- faithfulness
# --------------------------------------------------------------------------


def test_rule_decomposer_splits_sentences():
    claims = RuleDecomposer().decompose("A firm must report. It has 30 days.")
    assert claims == ["A firm must report.", "It has 30 days."]


def test_rule_decomposer_ignores_blank():
    assert RuleDecomposer().decompose("   ") == []


def test_faithfulness_fails_with_no_cited_text():
    res = check_faithfulness("Some answer.", "", RuleDecomposer(), StubNLIScorer())
    assert not res.passed
    assert res.reason == "no_cited_text"


def test_faithfulness_fails_with_no_claims():
    res = check_faithfulness("", "premise text", RuleDecomposer(), StubNLIScorer())
    assert not res.passed
    assert res.reason == "no_claims_extracted"


def test_faithfulness_passes_on_high_overlap():
    text = "An Authorised Person must retain records for six years."
    res = check_faithfulness(text, text, RuleDecomposer(), StubNLIScorer())
    assert res.passed
    assert res.min_entailment >= 0.5


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def _survivors(n_cand: int, n_ctrl: int) -> list[dict]:
    out = []
    for i in range(n_cand):
        out.append({"item_id": f"c{i}", "pool": CANDIDATE, "gold_passage_ids": ["g"], "q": i})
    for i in range(n_ctrl):
        out.append({"item_id": f"t{i}", "pool": CONTROL, "gold_passage_ids": ["g"], "q": i})
    return out


def test_sampling_respects_allocation():
    batch, report = stratified_sample(_survivors(500, 500), 100, 50)
    assert report["sampled_candidate"] == 100
    assert report["sampled_control"] == 50
    assert len(batch) == 150


def test_sampling_is_deterministic():
    a, _ = stratified_sample(_survivors(500, 500), 100, 50, seed=20260728)
    b, _ = stratified_sample(_survivors(500, 500), 100, 50, seed=20260728)
    assert [x["item_id"] for x in a] == [x["item_id"] for x in b]


def test_sampling_records_shortfall():
    _, report = stratified_sample(_survivors(30, 10), 100, 50)
    assert report["shortfall_candidate"] == 70
    assert report["shortfall_control"] == 40


def test_sampling_strips_gold_ids_from_batch():
    """The annotator must not see the gold set -- it would give the answer away."""
    batch, _ = stratified_sample(_survivors(20, 20), 10, 5)
    assert all("gold_passage_ids" not in item for item in batch)
    assert all("pool" not in item for item in batch)


def test_sampling_shuffles_pools_together():
    """If pools were not shuffled, position would leak pool membership."""
    batch, _ = stratified_sample(_survivors(100, 100), 50, 50)
    first_half = [i["_pool"] for i in batch[:50]]
    assert len(set(first_half)) == 2


def test_double_subset_size_and_membership():
    batch, _ = stratified_sample(_survivors(200, 200), 100, 50)
    sub = double_annotation_subset(batch, n=50)
    ids = {i["item_id"] for i in batch}
    assert len(sub) == 50
    assert all(i["item_id"] in ids for i in sub)


def test_double_subset_spans_both_pools():
    batch, _ = stratified_sample(_survivors(200, 200), 100, 50)
    sub = double_annotation_subset(batch, n=50)
    assert {i["_pool"] for i in sub} == {CANDIDATE, CONTROL}


# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------


def test_tokenize():
    assert tokenize("Category 2, Authorised!") == ["category", "2", "authorised"]


def test_bm25_finds_exact_match():
    corpus = {
        "p1": "capital adequacy ratio for a Category 2 licensed digital asset custodian",
        "p2": "financial promotions must be clear fair and not misleading",
        "p3": "record retention obligations of six years",
    }
    hits = BM25Retriever(corpus).search("capital adequacy Category 2 custodian", top_k=3)
    assert hits[0].passage_id == "p1"
    assert hits[0].rank == 1


def test_bm25_rejects_empty_corpus():
    with pytest.raises(ValueError):
        BM25Retriever({})


def test_rrf_promotes_consensus():
    a = [Hit("p1", 9.0, 1), Hit("p2", 8.0, 2)]
    b = [Hit("p2", 0.9, 1), Hit("p3", 0.8, 2)]
    fused = reciprocal_rank_fusion([a, b], top_k=3)
    assert fused[0].passage_id == "p2"
    assert [h.rank for h in fused] == [1, 2, 3]


# --------------------------------------------------------------------------
# Data adapter
# --------------------------------------------------------------------------


def test_adapter_handles_real_obliqa_record():
    """The exact shape documented in the ObliQA README."""
    q = adapt_record(
        {
            "QuestionID": "7824b4b4-bb50-432f-bc81-9f4cb80b2320",
            "Question": "What documentation does the FSRA require?",
            "Passages": [
                {"DocumentID": 30, "PassageID": "3)", "Passage": "INTRODUCTION. The framework..."},
                {"DocumentID": 30, "PassageID": "9)", "Passage": "DISCLOSURES. Rule 11.2.1..."},
            ],
            "Group": 3,
        }
    )
    assert q.question_id == "7824b4b4-bb50-432f-bc81-9f4cb80b2320"
    assert q.gold_passage_ids == ["30::3)", "30::9)"]


def test_passage_ids_are_document_scoped():
    """PassageID alone is only unique within a document.

    Regression guard: joining on the bare PassageID collides across ObliQA's 40
    documents, which would silently corrupt the S3 attribution screen -- the two
    passages below are different text under the same PassageID '3.1'.
    """
    a = adapt_record(
        {"QuestionID": "q1", "Question": "a",
         "Passages": [{"DocumentID": 11, "PassageID": "3.1", "Passage": "Doc 11 text."}]}
    )
    b = adapt_record(
        {"QuestionID": "q2", "Question": "b",
         "Passages": [{"DocumentID": 30, "PassageID": "3.1", "Passage": "Doc 30 text."}]}
    )
    assert a.gold_passage_ids != b.gold_passage_ids
    assert build_passage_corpus([a, b]) == {"11::3.1": "Doc 11 text.", "30::3.1": "Doc 30 text."}


def test_adapter_handles_capitalised_layout_without_docid():
    q = adapt_record(
        {
            "QuestionID": "q1",
            "Question": "What is the ratio?",
            "Passages": [{"PassageID": "p1", "Passage": "The ratio is 12%."}],
        }
    )
    assert q.question_id == "q1"
    assert q.gold_passage_ids == ["q1::p1"]


def test_adapter_handles_snake_case_layout():
    q = adapt_record(
        {
            "question_id": "q2",
            "question": "What?",
            "passages": [{"passage_id": "p9", "text": "Because."}],
        }
    )
    assert q.question_id == "q2"
    assert q.gold_passages[0].text == "Because."


def test_adapter_handles_bare_string_passages():
    q = adapt_record({"qid": "q3", "query": "Why?", "passages": ["some text"]})
    assert len(q.gold_passages) == 1
    assert q.gold_passages[0].passage_id == "q3::p0"


def test_adapter_raises_loudly_on_unknown_layout():
    with pytest.raises(ObliQAFormatError) as exc:
        adapt_record({"totally": "unexpected"})
    assert "Keys present" in str(exc.value)


def test_build_corpus_deduplicates_shared_passages():
    """The same ADGM passage cited by two questions appears once in the corpus."""
    rec = {"DocumentID": 11, "PassageID": "1.1.2", "Passage": "same"}
    a = adapt_record({"QuestionID": "q1", "Question": "a", "Passages": [rec]})
    b = adapt_record({"QuestionID": "q2", "Question": "b", "Passages": [rec]})
    assert build_passage_corpus([a, b]) == {"11::1.1.2": "same"}


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def test_render_prompt_includes_ids_and_question():
    prompt = render_prompt("What is X?", [("p1", "text one"), ("p2", "text two")])
    assert "[id: p1]" in prompt and "[id: p2]" in prompt
    assert "What is X?" in prompt


def test_stub_generator_output_passes_schema():
    prompt = render_prompt("What is X?", [("p1", "text one")])
    out = StubGenerator().generate([prompt])[0]
    res = check_schema(out)
    assert res.valid
    assert res.parsed["cited_passage_ids"] == ["p1"]


def test_stub_generator_is_batch_ordered():
    prompts = [render_prompt("q", [(f"p{i}", "t")]) for i in range(5)]
    outs = StubGenerator().generate(prompts)
    ids = [check_schema(o).parsed["cited_passage_ids"][0] for o in outs]
    assert ids == [f"p{i}" for i in range(5)]


def test_load_corpus_documents(tmp_path):
    """Full-corpus loader over the flat StructuredRegulatoryDocuments shape."""
    from saferag.data.obliqa import load_corpus_documents

    (tmp_path / "doc11.json").write_text(
        json.dumps(
            [
                {"ID": "x", "DocumentID": 11, "PassageID": "1.1.2", "Passage": "Where a Rule..."},
                {"ID": "y", "DocumentID": 11, "PassageID": "1.1.3", "Passage": "A Director..."},
            ]
        ),
        encoding="utf-8",
    )
    corpus = load_corpus_documents(tmp_path)
    assert corpus == {"11::1.1.2": "Where a Rule...", "11::1.1.3": "A Director..."}


def test_load_corpus_documents_missing_dir(tmp_path):
    from saferag.data.obliqa import load_corpus_documents

    with pytest.raises(FileNotFoundError):
        load_corpus_documents(tmp_path / "nope")
