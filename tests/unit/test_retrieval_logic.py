import pytest

from docqa.adapters.mongo.pipelines import hybrid_pipeline, text_only_pipeline, vector_only_pipeline
from docqa.adapters.rerank.cross_encoder import FastEmbedReranker
from docqa.adapters.rerank.noop import NoopReranker
from docqa.domain.errors import AccessFilterMissingError
from docqa.domain.models import AccessFilter, Chunk, RetrievedChunk
from docqa.retrieval.abstention import should_abstain
from docqa.retrieval.rrf import rrf_fuse
from docqa.settings import load_settings

CFG = load_settings(env_file=None).retrieval_cfg(vector_index="vi")
ACCESS = AccessFilter(owner_id="u1", doc_ids=["d1", "d2"])
ALLOWED_IN_BRANCH = {"$search", "$vectorSearch", "$match", "$sort", "$geoNear", "$limit"}


def hybrid(access=ACCESS):
    return hybrid_pipeline([0.1, 0.2], "refund window", access, CFG)


# ---------------------------------------------------------------- pipeline shape


def test_both_branches_prefilter_owner_and_documents():
    branches = hybrid()[0]["$rankFusion"]["input"]["pipelines"]
    vector = branches["vector"][0]["$vectorSearch"]
    text = branches["text"][0]["$search"]["compound"]
    assert vector["filter"] == {"owner_id": "u1", "doc_id": {"$in": ["d1", "d2"]}}
    assert {"equals": {"path": "owner_id", "value": "u1"}} in text["filter"]
    assert {"in": {"path": "doc_id", "value": ["d1", "d2"]}} in text["filter"]


def test_sub_pipelines_use_only_allowed_stages():
    # $limit is the only addition: it bounds the text branch, which has no limit of its own.
    for branch in hybrid()[0]["$rankFusion"]["input"]["pipelines"].values():
        assert {name for stage in branch for name in stage} <= ALLOWED_IN_BRANCH
        assert not any("$project" in stage for stage in branch)


def test_project_and_limits_come_after_fusion():
    stages = [next(iter(stage)) for stage in hybrid()]
    assert stages == ["$rankFusion", "$limit", "$project", "$addFields"]
    pipeline = hybrid()
    assert pipeline[1]["$limit"] == CFG.fused_k and pipeline[2]["$project"] == {"embedding": 0}
    branches = pipeline[0]["$rankFusion"]["input"]["pipelines"]
    assert branches["vector"][0]["$vectorSearch"]["limit"] == CFG.vector_k
    assert branches["text"][1] == {"$limit": CFG.text_k}


def test_weights_and_score_details_come_from_config():
    fusion = hybrid()[0]["$rankFusion"]
    assert fusion["combination"]["weights"] == {
        "vector": CFG.w_vector,
        "text": CFG.w_text,
    }
    assert fusion["scoreDetails"] is True


@pytest.mark.parametrize(
    "access",
    [
        None,
        AccessFilter.model_construct(owner_id="", doc_ids=["d"]),
        AccessFilter.model_construct(owner_id="u", doc_ids=[]),
    ],
)
def test_every_pipeline_refuses_an_empty_filter(access):
    with pytest.raises(AccessFilterMissingError):
        hybrid(access)
    with pytest.raises(AccessFilterMissingError):
        vector_only_pipeline([0.1], access, CFG)
    with pytest.raises(AccessFilterMissingError):
        text_only_pipeline("q", access, CFG)


def test_single_branch_pipelines_project_out_vectors():
    for pipeline in (
        vector_only_pipeline([0.1], ACCESS, CFG),
        text_only_pipeline("q", ACCESS, CFG),
    ):
        assert {"$project": {"embedding": 0}} in pipeline


# ---------------------------------------------------------------- rrf


def test_rrf_matches_a_hand_computed_example():
    fused = rrf_fuse({"vector": ["a", "b", "c"], "text": ["b", "d"]}, k=60)
    scores = {item.id: round(item.score, 6) for item in fused}
    assert scores == {
        "b": round(1 / 62 + 1 / 61, 6),
        "a": round(1 / 61, 6),
        "d": round(1 / 62, 6),
        "c": round(1 / 63, 6),
    }
    assert [item.id for item in fused] == ["b", "a", "d", "c"]
    assert fused[0].ranks == {"vector": 2, "text": 1}


def test_rrf_disjoint_lists_interleave():
    assert [i.id for i in rrf_fuse({"x": ["a", "b"], "y": ["c", "d"]})] == ["a", "c", "b", "d"]


def test_rrf_weights_favour_a_branch():
    fused = rrf_fuse({"vector": ["a"], "text": ["b"]}, weights={"vector": 2.0, "text": 1.0})
    assert [i.id for i in fused] == ["a", "b"] and fused[0].score == pytest.approx(2 / 61)


def test_rrf_empty():
    assert rrf_fuse({"vector": [], "text": []}) == []


# ---------------------------------------------------------------- re-rank and abstain


def retrieved(text: str, fused: float = 0.0, rerank: float | None = None) -> RetrievedChunk:
    chunk = Chunk(
        id=text,
        doc_id="d",
        owner_id="u",
        filename="f",
        text=text,
        page_start=1,
        page_end=1,
        content_hash="h",
    )
    return RetrievedChunk(chunk=chunk, fused_score=fused, rerank_score=rerank)


@pytest.mark.parametrize(
    "chunks, threshold, abstain",
    [
        ([], 0.0, True),
        ([retrieved("a", rerank=-1.0)], 0.0, True),
        ([retrieved("a", rerank=0.0)], 0.0, False),  # equal to the threshold answers
        ([retrieved("a", rerank=-3.0), retrieved("b", rerank=2.5)], 2.0, False),
        ([retrieved("a", fused=0.03)], 0.02, False),  # not re-ranked: fused score is used
    ],
)
def test_should_abstain(chunks, threshold, abstain):
    assert should_abstain(chunks, threshold) is abstain


class FakeCrossEncoder:
    def rerank(self, query, documents, batch_size=64):
        return [float(len(text)) for text in documents]  # longer text = more relevant


def test_cross_encoder_sorts_by_score_and_keeps_top_k(caplog):
    reranker = FastEmbedReranker("fake", max_tokens=2, model=FakeCrossEncoder())
    top = reranker.rerank("q", [retrieved("aa"), retrieved("aaaa"), retrieved("a")], top_k=2)
    assert [c.chunk.text for c in top] == ["aaaa", "aa"] and top[0].rerank_score == 4.0
    reranker.rerank("q", [retrieved("aaaaaaaaaa")], top_k=1)
    assert sum("truncated" in r.message for r in caplog.records) == 1  # warns once


def test_noop_reranker_keeps_fused_order():
    chunks = [retrieved("a"), retrieved("b"), retrieved("c")]
    assert NoopReranker().rerank("q", chunks, top_k=2) == chunks[:2]


def test_score_details_rank_na_becomes_none():
    # $rankFusion reports rank "NA" for a branch that did not return the chunk.
    from docqa.adapters.vectorstore.mongo_search import from_fused

    doc = {
        "_id": "c1",
        "doc_id": "d",
        "owner_id": "u",
        "filename": "f",
        "text": "t",
        "page_start": 1,
        "page_end": 1,
        "content_hash": "h",
        "fusion": {
            "value": 0.0164,
            "details": [
                {"inputPipelineName": "vector", "rank": 1, "weight": 1},
                {"inputPipelineName": "text", "rank": "NA", "weight": 1},
            ],
        },
    }
    result = from_fused(doc)
    assert (result.vector_rank, result.text_rank, result.fused_score) == (1, None, 0.0164)
