import datasyn as ds
import json
import pandas as pd
from scipy import stats


if __name__ == '__main__':
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.max_rows', None)

    with open('datasets/iris.json') as file:
        json_data = json.load(file)

    iris_data = pd.DataFrame(json_data)
    mushroom_data = pd.read_csv('datasets/mushrooms.csv')
    glassdoor_salary_full = pd.read_csv("datasets/Glassdoor Gender Pay Gap.csv")

    glassdoor_salary_full["Seniority"] = glassdoor_salary_full["Seniority"].astype(str)
    glassdoor_salary_full["PerfEval"] = glassdoor_salary_full["PerfEval"].astype(str)

    adult = pd.read_csv('datasets/adult.csv')
    # drop columns to clean data
    adult_cleaned = adult.drop(columns=["workclass", "occupation", "native.country",
                                        "fnlwgt", "capital.gain", "capital.loss"])

    adult_hours_pw = adult_cleaned["hours.per.week"]
    filtered_hpw = adult_hours_pw[adult_hours_pw <= 75]
    hpw_params = stats.norm.fit(filtered_hpw)

    cdg_adult = ds.get_cdg(adult_cleaned)
    cdg_iris = ds.get_cdg(iris_data)
    cdg_mushrooms = ds.get_cdg(mushroom_data)
    cdg_salaries_full = ds.get_cdg(glassdoor_salary_full)

    # ds.add_assoc_dist(cdg_iris, "petalWidth", "species", "setosa", "uniform", (0.1, 0.6))
    # ds.add_col_dist(cdg_iris, "petalWidth", "norm", (0.1, 0.25))
    # ds.mod_col_freq_dist(cdg_iris, "species", {"setosa": 0.4, "versicolor": 0.1, "virginica": 0.5})
    # ds.delete_assoc(cdg_iris, "petalLength", "species")

    ds.add_col_dist(cdg_adult, "hours.per.week", "norm", hpw_params)
    ucd_cdg_adult = ds.get_udc_top_order(cdg_adult)
    # ucd_cdg_iris = ds.get_udc_top_order(cdg_iris)

    new_iris_data = ds.generate_data(cdg_iris, 1000)
    new_mushroom_data = ds.generate_data(cdg_mushrooms, 1000)
    new_adult_data = ds.generate_data(ucd_cdg_adult, 1000)
    new_salary_data_full = ds.generate_data(cdg_salaries_full, 1000)

    # new_iris_data.to_csv("synthetic_datasets/iris_synthetic.csv", index=False)
    # new_mushroom_data.to_csv("synthetic_datasets/mushroom_synthetic.csv", index=False)
    # new_salary_data_full.to_csv("synthetic_datasets/salary_synthetic_full_new.csv", index=False)
    # new_adult_data.to_csv("synthetic_datasets/adult_synthetic_ucd.csv", index=False)
    # new_adult_data.to_csv("synthetic_datasets/adult_synthetic.csv", index=False)



