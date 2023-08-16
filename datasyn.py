import pandas as pd
import numpy as np
import math
from scipy import stats
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _get_frequency_dist(col: pd.Series) -> dict:
    frequency_dists = {}
    counts = col.value_counts(normalize=True)

    for att, frequency in counts.items():
        frequency_dists[att] = frequency

    return frequency_dists


def _get_min_max_bound(col: pd.Series) -> dict[str, float]:
    min_value = col.min()
    max_value = col.max()

    min_max_bounds = {"min": min_value, "max": max_value}

    return min_max_bounds


def _infer_distribution(column_data: pd.Series) -> [str, float]:
    distributions = ['norm', 'uniform', 'beta', 'expon', 'gamma']
    best_distribution = ''
    best_p_value = 0.0
    params_best_dist = ()

    for distribution in distributions:
        try:
            params = getattr(stats, distribution).fit(column_data)
        except stats.FitError:
            continue

        # perform KS test
        p_value = stats.kstest(column_data, distribution, args=params)[1]

        # check if this distribution provides a better fit
        if p_value > 0.05 and p_value > best_p_value:
            best_distribution = distribution
            best_p_value = p_value
            params_best_dist = params

    return best_distribution, best_p_value, params_best_dist


def _chi_square_result(source: pd.Series, target: pd.Series) -> float:
    significance_level = 0.05
    # count the frequencies using crosstab
    observed = np.array(pd.crosstab(source, target).values)
    stat, p_value, dof, expected = stats.chi2_contingency(observed)

    return p_value


def _conditional_entropy(source: pd.Series, target: pd.Series) -> float:
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


def _uncertainty_coefficient(x, y):
    h_xy = _conditional_entropy(x, y)
    x_counts = x.value_counts(normalize=True)
    h_x = stats.entropy(x_counts)

    if h_x == 0:
        return 1
    else:
        return (h_x - h_xy) / h_x


def _get_mean_dist_error(filtered_data: pd.DataFrame) -> float:
    distribution_errors = []
    cat_col = filtered_data.iloc[:, 0]
    cat_values = cat_col.unique()
    # print(f"cat values: {cat_values}")

    for value in cat_values:
        value_data = filtered_data.loc[cat_col == value].iloc[:, 1]

        distribution, p_value, params = _infer_distribution(value_data)

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
                "gen_node": True,
                "constraints": _get_frequency_dist(series)
            }
        else:
            # getting statistical min_max only for now, to-do: get distribution
            distribution, p_value, params = _infer_distribution(series)
            inf_error = 0
            is_gen_node = True

            if p_value > 0:
                inf_error = 1 - p_value

            if distribution == '':
                is_gen_node = False

            constraints[column] = {
                "type": "num",
                "gen_node": is_gen_node,
                "constraints": {
                    "min-max": _get_min_max_bound(series),
                    "distribution": distribution,
                    "params": params,
                    "inf_error": inf_error
                }
            }

    return constraints


def get_association_errors(input_data: pd.DataFrame) -> dict:
    association_errors = {}

    # create matrix for each source node A with target node B
    for column_A, series_A in input_data.items():
        # check whether column is categorical or numerical
        if series_A.dtype == object:
            if column_A not in association_errors:
                association_errors[column_A] = {}

            for column_B, series_B in input_data.items():
                if column_A == column_B:
                    continue
                # cat_cat
                elif series_B.dtype == object:
                    if column_B not in association_errors:
                        association_errors[column_B] = {}

                    print(f"check association between {column_A} and {column_B}...")
                    # perform chi-square test to determine whether there is a significant association
                    chi_square = _chi_square_result(series_A, series_B)
                    significance_level = 0.05
                    # measure strength of association with uncertainty coefficient
                    # in case p-value < significance level
                    if chi_square < significance_level:
                        u_value_ab = _uncertainty_coefficient(series_A, series_B)
                        u_value_ba = _uncertainty_coefficient(series_B, series_A)
                        if u_value_ab > u_value_ba:
                            association_errors[column_A][column_B] = 1 - u_value_ab
                        else:
                            association_errors[column_B][column_A] = 1 - u_value_ba
                    else:
                        print(f"no significant association between {column_A} and {column_B}")
                        continue
                # cat_num
                else:
                    filtered_data = input_data[[column_A, column_B]]
                    association_errors[column_A][column_B] = _get_mean_dist_error(filtered_data)

        else:
            # numerical columns as source nodes are not included
            continue

    return association_errors


def _get_max_edge(edges: dict) -> (str, str):
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


def _inverted_order_cdg(cdg: dict) -> dict:
    inverted_cdg = {}

    for n in cdg:
        inverted_cdg[n] = []
        for neighbor in cdg[n]["constraints"]["associations"]:
            if neighbor not in inverted_cdg:
                inverted_cdg[neighbor] = []

            inverted_cdg[neighbor].append(n)

    return inverted_cdg


def get_udc_top_order(cdg: dict) -> dict:
    inverted_order_cdg = _inverted_order_cdg(cdg)
    visited = set()
    ordering = []
    ordered_nodes = {}

    def dfs(n):
        visited.add(n)

        # check whether node has outgoing edges
        if n in inverted_order_cdg:
            for neighbor in inverted_order_cdg[n]:
                if neighbor not in visited:
                    dfs(neighbor)

        ordering.append(n)

    for node in inverted_order_cdg:
        if node not in visited:
            dfs(node)

    for n in ordering[::-1]:
        ordered_nodes[n] = cdg[n]

    return ordered_nodes


def _create_dag(graph: dict) -> ([], dict):
    new_graph = graph.copy()
    edges = dict()
    visited = set()
    ordering = []

    def dfs(n):
        visited.add(n)

        # check whether node has outgoing edges
        if n in new_graph:
            for neighbor in list(new_graph[n]):
                edges[n] = {}
                # print(f"trying to insert {graph[n][neighbor]} into edges[{n}][{neighbor}]")
                edges[n][neighbor] = new_graph[n][neighbor]
                if neighbor not in visited:
                    dfs(neighbor)
                else:
                    # if node has already been visited
                    # check whether neighbor has an outgoing edge
                    if neighbor in new_graph:
                        source, target = _get_max_edge(edges)
                        # print(f"deleting new_graph[{source}][{target}]")
                        if new_graph[source][target] in new_graph:
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


def _get_conditional_dist_cat(source: pd.Series, target: pd.Series) -> dict:
    contingency_table = pd.crosstab(source, target)
    conditional_probs = contingency_table.div(contingency_table.sum(axis=0), axis=1)
    print(conditional_probs)
    return conditional_probs.to_dict()


def _get_conditional_dist_num(source: pd.Series, target: pd.Series, input_data: pd.DataFrame) -> dict:
    conditional_dist = {}
    grouped_data = input_data.groupby(source)[target]

    for source_value, target_col in grouped_data:
        target_series = target_col.squeeze()
        distribution, p_value, params = _infer_distribution(target_series)
        conditional_dist[source_value] = {
            "distribution": distribution,
            "params": params
        }

    return conditional_dist


def _get_conditional_min_max_bound(source: pd.Series, target: pd.Series) -> dict:
    df = pd.DataFrame({'source': source, 'target': target})
    min_max = df.groupby('source')['target'].agg(['min', 'max'])
    print(min_max)

    return min_max.to_dict(orient="index")


def _invert_dag(dag: dict) -> dict:
    inverted_dag = {}

    for n, neighbors in dag.items():
        for neighbor, weight in neighbors.items():
            if neighbor not in inverted_dag:
                inverted_dag[neighbor] = {}

            inverted_dag[neighbor][n] = weight

    return inverted_dag


def get_association_constraints(input_data: pd.DataFrame) -> dict:
    association_cons = {}
    association_errors = get_association_errors(input_data)
    association_dag = _create_dag(association_errors)[1]
    inverted_association_errors = _invert_dag(association_dag)

    for column_A in inverted_association_errors:
        association_cons[column_A] = {}
        for column_B in inverted_association_errors[column_A]:
            if input_data[column_A].dtype == object:
                association_cons[column_A][column_B] = {
                    "inf_error": inverted_association_errors[column_A][column_B],
                    "conditional_dist": _get_conditional_dist_cat(input_data[column_A], input_data[column_B])
                }
            else:
                association_cons[column_A][column_B] = {
                    "inf_error": inverted_association_errors[column_A][column_B],
                    "min-max": _get_conditional_min_max_bound(input_data[column_B], input_data[column_A]),
                    "distributions": _get_conditional_dist_num(column_B, column_A, input_data)
                }

    return association_cons


def get_cdg(input_data: pd.DataFrame) -> dict:
    column_constraints = get_column_constraints(input_data)
    assoc_errors = get_association_errors(input_data)
    top_order, dag = _create_dag(assoc_errors)
    association_constraints = get_association_constraints(input_data)

    nodes = {}
    ordered_nodes = {}

    for n, value in column_constraints.items():
        nodes[n] = {}
        association_cons = {}
        targets = []

        if n in dag:
            targets = list(dag[n].keys())

        if n in association_constraints:
            association_cons = association_constraints[n]

        nodes[n] = {
            "type": column_constraints[n]["type"],
            "gen_node": column_constraints[n]["gen_node"],
            "constraints": {
                "column": column_constraints[n]["constraints"],
                # all nodes pointing to n
                "associations": association_cons
            },
            "targets": targets
        }

    for n in top_order:
        ordered_nodes[n] = nodes[n]

    return ordered_nodes


def _generate_num_column(min_bound: int, max_bound: int, distribution: str, params: tuple, n: int) -> np.array:
    if distribution == '':
        return np.random.uniform(min_bound, max_bound, n)
    else:
        return getattr(stats, distribution).rvs(*params, size=n)


def add_col_dist(cdg: dict, column: str, distribution: str, params: tuple):
    if cdg[column]["type"] != "num":
        raise TypeError(f"Column {column} is not of type num.")

    if distribution in ['norm', 'uniform', 'beta', 'expon', 'gamma']:
        cdg[column]["constraints"]["column"]["distribution"] = distribution
    else:
        raise ValueError("No valid distribution given. Try 'norm', 'uniform', 'beta', 'expon' or 'gamma'")

    # todo: check validity of the tuple matching distribution
    cdg[column]["constraints"]["column"]["params"] = params

    for association in list(cdg[column]["constraints"]["associations"]):
        del cdg[column]["constraints"]["associations"][association]


def add_assoc_dist(cdg: dict, col_A: str, col_B: str, val_B: str, distribution: str, params: tuple):
    if cdg[col_A]["type"] != "num":
        raise TypeError(f"Column {col_A} is not of type num.")

    if cdg[col_B]["type"] != "cat":
        raise TypeError(f"Column {col_B} is not of type cat.")

    if distribution in ['norm', 'uniform', 'beta', 'expon', 'gamma']:
        cdg[col_A]["constraints"]["associations"][col_B]["distributions"][val_B]["distribution"] = distribution
    else:
        raise ValueError("No valid distribution given. Try 'norm', 'uniform', 'beta', 'expon' or 'gamma'")

    # todo: check validity of the tuple matching distribution
    cdg[col_A]["constraints"]["associations"][col_B]["distributions"][val_B]["params"] = params

    for association in list(cdg[col_A]["constraints"]["associations"]):
        if association == col_B:
            continue
        else:
            del cdg[col_A]["constraints"]["associations"][association]


def mod_col_freq_dist(cdg: dict, column: str, new_frequencies: dict):
    if cdg[column]["type"] != "cat":
        raise TypeError(f"Column {column} is not of type cat.")

    if sum(new_frequencies.values()) != 1.0:
        raise ValueError("New frequencies have to add up to 1.0")

    cdg[column]["constraints"]["column"] = new_frequencies

    for association in list(cdg[column]["constraints"]["associations"]):
        del cdg[column]["constraints"]["associations"][association]


def mod_assoc_freq_dist(cdg: dict, col_A: str, col_B: str, val_B: str, new_frequencies: dict):
    if cdg[col_A]["type"] != "cat" and cdg[col_B]["type"] != "cat":
        raise TypeError("Both columns need to be of type cat.")

    if sum(new_frequencies.values()) != 1.0:
        raise ValueError("New frequencies have to add up to 1.0")

    cdg[col_A]["constraints"]["associations"][col_B]["conditional_dist"][val_B] = new_frequencies

    for association in list(cdg[col_A]["constraints"]["associations"]):
        if association == col_B:
            continue
        else:
            del cdg[col_A]["constraints"]["associations"][association]


def delete_assoc(cdg: dict, column_A: str, column_B: str):
    if column_B in cdg[column_A]["constraints"]["associations"]:
        del cdg[column_A]["constraints"]["associations"][column_B]
    else:
        raise KeyError(f"Column {column_A} with source node {column_B} does not exist inside the CDG.")


def _get_min_edge(association_constraints: dict) -> str:
    min_inf_error = 1.1
    incoming_node = ''
    for association in association_constraints:
        inf_error = association_constraints[association]["inf_error"]
        if inf_error < min_inf_error:
            min_inf_error = inf_error
            incoming_node = association

    return incoming_node


def generate_data(cdg: dict, n: int) -> pd.DataFrame:
    gen_data = {}

    for column, details in cdg.items():
        column_constraints = details["constraints"]["column"]
        association_constraints = details["constraints"]["associations"]
        col_type = details["type"]

        # case 1: no incoming edges
        if len(association_constraints) == 0:
            if col_type == "cat":
                values = list(column_constraints.keys())
                p = list(column_constraints.values())
                gen_data[column] = np.random.choice(values, n, p=p)
            else:
                min_bound = column_constraints["min-max"]["min"]
                max_bound = column_constraints["min-max"]["max"]
                distribution = column_constraints["distribution"]
                params = column_constraints["params"]
                gen_data[column] = _generate_num_column(min_bound, max_bound, distribution, params, n)

        # case 2: one incoming edge
        elif len(association_constraints) == 1:
            incoming_node = list(association_constraints.keys())[0]

            if incoming_node in gen_data:
                column_data = []

                if col_type == "cat":
                    # iterate through the values of the generated source column
                    for source_value in gen_data[incoming_node]:
                        # generate value based on the conditional probabilities
                        conditional_dist = association_constraints[incoming_node]["conditional_dist"][source_value]
                        target_values = list(conditional_dist.keys())
                        target_p = list(conditional_dist.values())
                        column_data.append(np.random.choice(target_values, p=target_p))

                # numerical column
                else:
                    for source_value in gen_data[incoming_node]:
                        source_value_dist = association_constraints[incoming_node]["distributions"][source_value]
                        distribution = source_value_dist["distribution"]
                        params = source_value_dist["params"]
                        min_bound = association_constraints[incoming_node]["min-max"][source_value]["min"]
                        max_bound = association_constraints[incoming_node]["min-max"][source_value]["max"]

                        if distribution == '':
                            column_data.append(np.random.uniform(min_bound, max_bound))
                        else:
                            column_data.append(getattr(stats, distribution).rvs(*params))

                gen_data[column] = column_data

            else:
                raise KeyError(f"column {incoming_node} has not been generated yet")

        # case 3: multiple incoming edges
        else:
            incoming_node = _get_min_edge(association_constraints)

            if incoming_node in gen_data:
                column_data = []

                if col_type == "cat":
                    for source_value in gen_data[incoming_node]:
                        conditional_dist = association_constraints[incoming_node]["conditional_dist"][source_value]
                        target_values = list(conditional_dist.keys())
                        target_p = list(conditional_dist.values())
                        column_data.append(np.random.choice(target_values, p=target_p))

                    gen_data[column] = column_data

                else:
                    for source_value in gen_data[incoming_node]:
                        source_value_dist = association_constraints[incoming_node]["distributions"][source_value]
                        distribution = source_value_dist["distribution"]
                        params = source_value_dist["params"]
                        min_bound = association_constraints[incoming_node]["min-max"][source_value]["min"]
                        max_bound = association_constraints[incoming_node]["min-max"][source_value]["max"]

                        if distribution == '':
                            column_data.append(np.random.uniform(min_bound, max_bound))
                        else:
                            column_data.append(getattr(stats, distribution).rvs(*params))

                    gen_data[column] = column_data

    return pd.DataFrame(gen_data)

