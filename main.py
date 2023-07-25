from collections import defaultdict

import pandas as pd
import numpy as np
import math
from scipy import stats
import json
import networkx as nx


def get_frequency_dist(col: pd.Series) -> dict:
    frequency_dists = {}
    counts = col.value_counts(normalize=True)

    for att, frequency in counts.items():
        frequency_dists[att] = frequency

    return frequency_dists


def get_min_max_bound(col: pd.Series) -> dict[str, float]:
    min_value = col.min()
    max_value = col.max()

    min_max_bounds = {"min": min_value, "max": max_value}

    return min_max_bounds


def infer_distribution(column_data: pd.Series) -> [str, float]:
    distributions = ['norm', 'uniform', 'beta', 'expon', 'gamma']
    best_distribution = ''
    best_p_value = 0.0

    for distribution in distributions:

        try:
            params = getattr(stats, distribution).fit(column_data)
        except stats._warnings_errors.FitError:
            # Handle the FitError here
            print(f"FitError occurred for value.")
            # Perform alternative actions or set default parameter values

        # fit the distribution to data
        # params = getattr(stats, distribution).fit(column_data)

        # perform KS test
        p_value = stats.kstest(column_data, distribution, args=params)[1]

        # check if this distribution provides a better fit
        if p_value > 0.05 and p_value > best_p_value:
            best_distribution = distribution
            best_p_value = p_value

    return best_distribution, best_p_value


def chi_square_result(source: pd.Series, target: pd.Series) -> float:
    # count the frequencies using crosstab
    observed = np.array(pd.crosstab(source, target).values)

    # degrees of freedom: (number of rows - 1) * (number of columns - 1)
    ddof = (len(observed) - 1) * (len(observed[0]) - 1)
    p_value = stats.chisquare(observed, ddof=ddof, axis=None)[1]

    return p_value


def conditional_entropy(source: pd.Series, target: pd.Series) -> float:
    counts = pd.crosstab(source, target)
    total_occurrences = counts.sum().sum()
    entropy = 0

    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            p_xy = counts.iloc[i, j] / total_occurrences
            p_y = counts.iloc[:, j].sum() / total_occurrences
            if p_xy > 0:
                entropy += p_xy * math.log(p_y / p_xy)

    return entropy


def theil_u(x, y):
    h_xy = conditional_entropy(x, y)
    x_counts = x.value_counts(normalize=True)
    h_x = stats.entropy(x_counts)

    if h_x == 0:
        return 1
    else:
        return (h_x - h_xy) / h_x


def get_mean_dist_error(filtered_data: pd.DataFrame) -> float:
    distribution_errors = []
    cat_col = filtered_data.iloc[:, 0]
    cat_values = cat_col.unique()
    # print(f"cat values: {cat_values}")

    for value in cat_values:
        value_data = filtered_data.loc[cat_col == value].iloc[:, 1]

        # try:
        #     params = getattr(stats, distribution).fit(value_data)
        # except stats._warnings_errors.FitError:
        #     print(f"FitError occurred for value {value}.")
        #
        # p_value = stats.kstest(value_data, distribution, args=params)[1]

        distribution, p_value = infer_distribution(value_data)

        distribution_errors.append(1 - p_value)

    # print(f"distribution errors: {distribution_errors}")
    mean_dist_error = sum(distribution_errors) / len(distribution_errors)

    return mean_dist_error


# infer column constraints from training data
def get_column_constraints(input_data: pd.DataFrame) -> dict:
    constraints = {}

    for column, series in input_data.items():
        # check whether column is categorical or numerical
        if series.dtype == object:
            constraints[column] = {
                "type": "cat",
                "constraints": get_frequency_dist(series)
            }
        else:
            # getting statistical min_max only for now, to-do: get distribution
            distribution, p_value = infer_distribution(series)
            inf_error = 0
            if p_value > 0:
                inf_error = 1 - p_value
            constraints[column] = {
                "type": "num",
                "constraints": {
                    "min-max": get_min_max_bound(series),
                    "distribution": distribution,
                    "inf_error": inf_error
                }
            }

    return constraints


def get_association_constraints(input_data: pd.DataFrame) -> dict:
    association_constraints = {}

    # create matrix for each source node A with target node B
    for column_A, series_A in input_data.items():
        # check whether column is categorical or numerical
        if series_A.dtype == object:
            association_constraints[column_A] = {}

            for column_B, series_B in input_data.items():
                if column_A == column_B:
                    continue
                # cat_cat
                elif series_B.dtype == object:
                    print(f"check association between {column_A} and {column_B}...")
                    # perform chi-square test to determine whether there is a significant association
                    chi_square = chi_square_result(series_A, series_B)
                    significance_level = 0.05
                    # measure strength of association with uncertainty coefficient
                    # in case p-value < significance level
                    if chi_square < significance_level:
                        uncertainty_coefficient = theil_u(series_A, series_B)
                        association_constraints[column_A][column_B] = 1 - uncertainty_coefficient
                    else:
                        print(f"no significant association between {column_A} and {column_B}")
                        continue
                # cat_num
                else:
                    filtered_data = input_data[[column_A, column_B]]
                    association_constraints[column_A][column_B] = get_mean_dist_error(filtered_data)

        else:
            # numerical columns as source nodes are not included
            continue

    return association_constraints


def reduce_matrix(graph: dict) -> dict:
    new_graph = graph.copy()
    visited = set()

    def dfs(n):
        visited.add(n)
        if n in graph:
            for neighbor in list(new_graph[n]):
                if neighbor not in visited:
                    dfs(neighbor)
                else:
                    # if cycle is detected
                    if neighbor in new_graph and new_graph[neighbor].get(n) is not None:
                        if new_graph[n][neighbor] < new_graph[neighbor][n]:
                            del new_graph[neighbor][n]
                        else:
                            del new_graph[n][neighbor]

    for node in new_graph:
        if node not in visited:
            dfs(node)

    return new_graph


def get_max_edge(edges: dict) -> (str, str):
    max_key = None
    max_inner_key = None
    max_value = float('-inf')

    for key, inner_dict in edges.items():
        for inner_key, value in inner_dict.items():
            if value > max_value:
                max_key = key
                max_inner_key = inner_key
                max_value = value

    return max_key, max_inner_key


def create_dag(graph: dict) -> ([], dict):
    new_graph = graph.copy()
    edges = dict()
    visited = set()
    ordering = []

    def dfs(n):
        visited.add(n)

        # check whether node has outgoing edges
        if n in graph:
            for neighbor in list(new_graph[n]):
                edges[n] = {}
                # print(f"trying to insert {graph[n][neighbor]} into edges[{n}][{neighbor}]")
                edges[n][neighbor] = graph[n][neighbor]
                if neighbor not in visited:
                    dfs(neighbor)
                else:
                    # if node has already been visited
                    # check whether neighbor has an outgoing edge
                    if neighbor in graph:
                        source, target = get_max_edge(edges)
                        # print(f"deleting new_graph[{source}][{target}]")
                        del new_graph[source][target]

        # if node has no outgoing edges
        else:
            print(f"edges: {edges} cleared")
            edges.clear()

        ordering.append(n)

    for node in new_graph:
        if node not in visited:
            dfs(node)

    return ordering[::-1], new_graph


# 'merge' inferred constraints with user defined constraints
# columns existent in the udc simply override the current inferred constraints
def get_merged_constraints(user_def_cons: dict, inferred_cons: dict) -> dict:
    merged_constraints = {}

    for column, constraint in inferred_cons.items():
        if column in user_def_cons:
            merged_constraints[column] = user_def_cons[column]
        else:
            merged_constraints[column] = inferred_cons[column]

    return merged_constraints


def generate_num_column(min_bound: int, max_bound: int, distribution: str, n: int) -> np.array:
    interval_size = max_bound - min_bound

    if distribution == "norm":
        # make use of 68–95–99.7 rule
        mean = (min_bound + max_bound) / 2
        std = mean / 3

        values = np.random.normal(mean, std, n)

        return np.clip(values, min_bound, max_bound)

    elif distribution == "uniform":
        return np.random.uniform(min_bound, max_bound, n)

    elif distribution == "gamma":
        values = np.random.gamma(shape=2, scale=2, size=n)

        # normalize values to the range [0, 1]
        normalized_values = (values - np.min(values)) / (np.max(values) - np.min(values))

        # scale the normalized values to the desired bounds
        return min_bound + (interval_size * normalized_values)

    elif distribution == "beta":
        # beta distribution generates values in the range [0, 1]
        values = np.random.beta(a=2, b=3, size=n)

        # scale values to the desired bounds
        return min_bound + (interval_size * values)

    elif distribution == "expon":
        values = np.random.exponential(scale=2, size=n)
        return min_bound + (interval_size * values / np.max(values))

    else:
        return np.random.randint(min_bound, max_bound + 1, n)


def generate_data_v1(constraints: dict, n: int) -> pd.DataFrame:
    gen_data = {}

    for column, constraint in constraints.items():
        cons = constraint["constraints"]

        if constraint["type"] == "cat":
            values = list(cons.keys())
            p = list(cons.values())
            gen_data[column] = np.random.choice(values, n, p=p)

        elif constraint["type"] == "num":
            min_bound = cons["min-max"]["min"]
            max_bound = cons["min-max"]["max"]
            distribution = cons["distribution"]
            gen_data[column] = generate_num_column(min_bound, max_bound, distribution, n)

        else:
            raise KeyError(f"Expected key 'type' not found.")

    return pd.DataFrame(gen_data)


if __name__ == '__main__':
    data = pd.DataFrame({
        'Fruit': ['Apple', 'Apple', 'Apple', 'Banana', 'Banana', 'Banana', 'Orange', 'Orange', 'Orange'],
        'Sweetness Level': ['High', 'High', 'Medium', 'Medium', 'Medium', 'High', 'High', 'Medium', 'High'],
        'Quantity Sold': [90, 100, 80, 110, 140, 130, 120, 90, 100]
    })

    data_2 = pd.DataFrame({
        'Gender': ['Male', 'Female', 'Male', 'Male', 'Female', 'Male', 'Female', 'Male', "Female"],
        'Income': [50000, 60000, 35000, 70000, 55000, 48000, 30000, 70000, 30000],
        'Education': ['High School', "High School", "Associate's", "Master's", 'High School', "Bachelor's",
                      "High School", "Bachelor's", "High School"]
    })

    user_defined_cons = {
        # modify feature constraint; change gender to 1:1 ratio
        'gender': {'type': 'cat', 'constraints': {'Male': 0.5, 'Female': 0.5}},
        # modify feature constraint; change the min-max bound
        # add feature constraint; add normal distribution to numerical column
        'income': {'type': 'num', 'constraints': {'min-max': {'min': 20000, 'max': 55000}, 'distribution': 'norm'}}
    }

    # Open the JSON file
    with open('glassdoor_salaries.json') as file:
        json_data = json.load(file)

    # Convert JSON data to a DataFrame
    data_3 = pd.DataFrame(json_data)

    col_cons = get_column_constraints(data_3)
    print("column constraints:")
    print(col_cons)
    assoc_cons = get_association_constraints(data_3)
    print("association constraints:")
    print(assoc_cons)
    reduced_matrix = reduce_matrix(assoc_cons)
    dag = create_dag(reduced_matrix)
    print(f"dag: {dag}")
    new_data = generate_data_v1(col_cons, 10)
    print("new data:")
    print(new_data)
    merged_cons = get_merged_constraints(user_defined_cons, col_cons)
    print("merged constraints")
    print(merged_cons)
    new_data_udc = generate_data_v1(merged_cons, 20)
    print("new data with merged constraints:")
    print(new_data_udc)


#################################
# backlog
#################################

def create_cdg(constraints):
    pass


def data_syn_with_cons(cdg, num_samples):
    pass


def data_synthesis(input_data, num_samples):
    constraints = get_column_constraints(input_data)
    cdg = create_cdg(constraints)
    return data_syn_with_cons(cdg, num_samples)
