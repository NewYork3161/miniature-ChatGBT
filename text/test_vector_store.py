import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vector_store import VectorStore

def test_vector_store_search():
    store = VectorStore()

    store.add([1, 0, 0], "A")
    store.add([0, 1, 0], "B")

    results = store.search([1, 0, 0], top_k=1)

    assert results[0] == "A"