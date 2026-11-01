"""Exercise real PDF parsing/retrieval and the public API without a model."""
from paper_rag.api import QueryRequest, health, query


def test_real_pdf_query_and_citation_contract():
    response=query(QueryRequest(question='retrieval augmented generation',top_k=2))
    assert health()['chunks']>0
    assert response['sources'] and response['retrieval_backend']=='tfidf-cpu'
    assert response['generation'] is None
    assert response['citation_check']['necessary_coverage'] is None
    assert response['citation_check']['support_status']=='requires_human_review'
