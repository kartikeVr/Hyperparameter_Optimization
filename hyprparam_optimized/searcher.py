import random 

class GridSearch:
    def __init__(self,params):
        self.params = params


    def generate(self):
        keys = list(self.params.keys())
        combinations = [{}]

        for key in keys:
            new_combinations =[]
            for combo in combinations:
                for value in self.params[key]:
                    new_combo = combo.copy()
                    new_combo[key] = value
                    new_combinations.append(new_combo)
            combinations = new_combinations
        return combinations

class RandomSearch:
    def __init__(self,params,n_iter,random_state=None):
        self.params = params 
        self.n_iter = n_iter
        if random_state is not None:
            random.seed(random_state)
            
    def generate(self):
        keys = list(self.params.keys())
        samples = []

        for _ in range(self.n_iter):
            params = {}
            for key in keys:
                params[key] = random.choice(self.params[key])
            samples.append(params)

        return samples