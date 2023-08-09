from sdv.evaluation.single_table import evaluate_quality
from sdv.evaluation.single_table import run_diagnostic
from sdv.metadata import SingleTableMetadata
import pandas as pd
import json

if __name__ == '__main__':
    with open('datasets/iris.json') as file:
        json_data = json.load(file)

    with open('datasets/glassdoor_salaries.json') as file:
        salaries_json = json.load(file)

    iris_data = pd.DataFrame(json_data)
    mushroom_data = pd.read_csv('datasets/mushrooms.csv')
    salary_data = pd.DataFrame(salaries_json)
    adult = pd.read_csv('datasets/adult.csv')
    adult_data = adult.drop(columns=["workclass", "fnlwgt", "occupation", "hours.per.week", "native.country"])

    iris_syn = pd.read_csv("synthetic_datasets/iris_synthetic.csv")
    adult_syn = pd.read_csv('synthetic_datasets/adult_synthetic.csv')
    mushroom_syn = pd.read_csv("synthetic_datasets/mushroom_synthetic.csv")
    salary_syn = pd.read_csv("synthetic_datasets/salary_synthetic.csv")

    iris_meta = SingleTableMetadata()
    iris_meta.detect_from_dataframe(data=iris_syn)
    adult_meta = SingleTableMetadata()
    real_adult_meta = SingleTableMetadata()
    adult_meta.detect_from_dataframe(data=adult_syn)
    real_adult_meta.detect_from_dataframe(data=adult_data)
    mushroom_meta = SingleTableMetadata()
    mushroom_meta.detect_from_dataframe(data=mushroom_syn)
    salary_meta = SingleTableMetadata()
    salary_meta.detect_from_dataframe(data=salary_syn)

    print("Iris report:")
    iris_quality_report = evaluate_quality(iris_data, iris_syn, iris_meta)
    iris_diagnostic = run_diagnostic(iris_data, iris_syn, iris_meta)

    # print("Adult report:")
    # adult_quality_report = evaluate_quality(adult_data, adult_syn, adult_meta)
    # adult_diagnostic = run_diagnostic(adult_data, adult_syn, adult_meta)

    print("Mushroom report:")
    mushroom_quality_report = evaluate_quality(mushroom_data, mushroom_syn, mushroom_meta)
    mushroom_diagnostic = run_diagnostic(mushroom_data, mushroom_syn, mushroom_meta)

    print("Salary report:")
    salary_quality_report = evaluate_quality(salary_data, salary_syn, salary_meta)
    salary_diagnostic = run_diagnostic(salary_data, salary_syn, salary_meta)
