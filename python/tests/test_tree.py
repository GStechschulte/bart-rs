import numpy as np




depth = 10
value_tree = np.zeros(2 ** depth)
print(value_tree.shape)


rules = {0: "ContinuousSplit", 1: "Hey", 2 :"ContinuousSplit"}
required = set(range(3))
passed = set(list(rules.keys()))

if not required.issubset(passed):
    missing = sorted(list(required - passed))
    raise ValueError(f"Missing: {missing}")

supported_split_rules = {"ContinuousSplit", "OneHotSplit", "SubsetSplit"}
passed_rules = set(rules.values())
invalid_rules = passed_rules - supported_split_rules

if invalid_rules:
    raise ValueError(
        f"rule(s) must be one of {supported_split_rules}. Received invalid rule(s): {invalid_rules}"
    )
