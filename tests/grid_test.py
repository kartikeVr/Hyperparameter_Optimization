from hyprparam_optimized.searcher import GridSearch

def test_grid_search_combination_count():
    param_space = {
        "a": [1, 2],
        "b": [10, 20, 30]
    }

    grid = GridSearch(param_space)
    combos = grid.generate()

    assert len(combos) == 6
