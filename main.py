import datasyn as ds
import json
import pandas as pd

if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    with open('datasets/iris.json') as file:
        json_data = json.load(file)

    with open('datasets/glassdoor_salaries.json') as file:
        salaries_json = json.load(file)

    iris_data = pd.DataFrame(json_data)
    mushroom_data = pd.read_csv('datasets/mushrooms.csv')
    glassdoor_salary_data = pd.DataFrame(salaries_json)
    adult = pd.read_csv('datasets/adult.csv')
    adult_cleaned = adult.drop(columns=["workclass", "fnlwgt", "occupation", "hours.per.week", "native.country"])

    cdg_adult = ds.get_cdg(adult_cleaned)
    cdg_iris = ds.get_cdg(iris_data)
    cdg_mushrooms = ds.get_cdg(mushroom_data)
    cdg_salaries = ds.get_cdg(glassdoor_salary_data)

    # ds.add_assoc_dist(cdg, "petalWidth", "species", "setosa", "uniform", (0.1, 0.6))
    # ds.add_col_dist(cdg, "petalWidth", "norm", (0.1, 0.25))
    # ds.mod_col_freq_dist(cdg, "species", {"setosa": 0.4, "versicolor": 0.1, "virginica": 0.5})
    # ds.delete_assoc(cdg, "sepalLength", "species")
    new_iris_data = ds.generate_data(cdg_iris, 1000)
    new_mushroom_data = ds.generate_data(cdg_mushrooms, 1000)
    new_salary_data = ds.generate_data(cdg_salaries, 1000)
    new_adult_data = ds.generate_data(cdg_adult, 1000)

    new_iris_data.to_csv("synthetic_datasets/iris_synthetic.csv")
    new_mushroom_data.to_csv("synthetic_datasets/mushroom_synthetic.csv")
    new_salary_data.to_csv("synthetic_datasets/salary_synthetic.csv")
    new_adult_data.to_csv("synthetic_datasets/adult_synthetic.csv")


