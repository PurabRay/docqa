import pytest

from docqa.adapters.mongo.pipelines import text_count_pipeline, vector_count_pipeline
from docqa.domain.errors import AccessFilterMissingError, StorageCapReachedError
from docqa.domain.models import AccessFilter
from docqa.guardrails.storage_guard import StorageGuard, check_capacity
from docqa.ports.storage import StorageUsage

ACCESS = AccessFilter(owner_id="u1", doc_ids=["d1", "d2"])


@pytest.mark.parametrize(
    "size_mb, documents, ok",
    [(10, 1, True), (399.9, 499, True), (400, 1, False), (10, 500, False), (512, 600, False)],
)
def test_capacity_thresholds(size_mb, documents, ok):
    usage = StorageUsage(size_mb=size_mb, documents=documents)
    if ok:
        check_capacity(usage, cap_mb=400, max_documents=500)
    else:
        with pytest.raises(StorageCapReachedError):
            check_capacity(usage, cap_mb=400, max_documents=500)


async def test_guard_reads_the_meter():
    class Meter:
        async def usage(self):
            return StorageUsage(size_mb=450, documents=3)

    with pytest.raises(StorageCapReachedError, match="450 MB"):
        await StorageGuard(Meter(), cap_mb=400, max_documents=500).ensure_capacity()


def test_text_count_filters_owner_and_documents():
    [stage] = text_count_pipeline(ACCESS, "chunks_text")
    filters = stage["$searchMeta"]["compound"]["filter"]
    assert {"equals": {"path": "owner_id", "value": "u1"}} in filters
    assert {"in": {"path": "doc_id", "value": ["d1", "d2"]}} in filters


def test_vector_count_filters_owner_and_documents():
    search = vector_count_pipeline([0.1, 0.2], ACCESS, "chunks_vector", limit=7)[0]["$vectorSearch"]
    assert search["filter"] == {"owner_id": "u1", "doc_id": {"$in": ["d1", "d2"]}}
    assert search["limit"] == 7 and search["exact"] is True


def test_pipelines_refuse_a_missing_access_filter():
    with pytest.raises(AccessFilterMissingError):
        text_count_pipeline(None, "chunks_text")
    with pytest.raises(AccessFilterMissingError):
        vector_count_pipeline([0.1], None, "chunks_vector", limit=1)
