import pytest
from restaurant_reviews_allergy.review_parser.allergen_search import simple_allergen_search


@pytest.mark.parametrize("text,allergen", [('cat', 'at'), ('dog', 'og')])
def test_simple_allergen_search_true(text, allergen):
    assert simple_allergen_search(text, allergen) == True

@pytest.mark.parametrize("text,allergen", [('cat', 'og'), ('dog', 'at')])
def test_simple_allergen_search_false(text, allergen):
    assert simple_allergen_search(text, allergen) == False
