from hyprparam_optimized.searcher import RandomSearch

def test_random_search_iterations():
    param_space = {
        "x": [1, 2, 3, 4]
    }

    rs = RandomSearch(param_space, n_iter=5, random_state=42)
    samples = rs.generate()

    assert len(samples) == 5
