from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata
import pandas as pd
import json

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)

    with open('datasets/iris.json') as file:
        json_data = json.load(file)

    with open('datasets/glassdoor_salaries.json') as file:
        salaries_json = json.load(file)

    iris_data = pd.DataFrame(json_data)
    mushroom_data = pd.read_csv('datasets/mushrooms.csv')
    salary_data = pd.read_csv("datasets/Glassdoor Gender Pay Gap.csv")
    salary_cleaned = pd.read_csv("datasets/salary_converted.csv")
    adult = pd.read_csv('datasets/adult.csv')
    adult_data = adult.drop(columns=["workclass", "occupation", "native.country"])

    adult_cleaned = adult.drop(columns=["workclass", "occupation", "native.country",
                                        "fnlwgt", "capital.gain", "capital.loss"])

    iris_syn = pd.read_csv("synthetic_datasets/iris_synthetic.csv")
    adult_syn = pd.read_csv('synthetic_datasets/adult_synthetic.csv')
    mushroom_syn = pd.read_csv("synthetic_datasets/mushroom_synthetic.csv")
    salary_syn = pd.read_csv("synthetic_datasets/salary_synthetic_full.csv")

    salary_cleaned_syn = pd.read_csv("synthetic_datasets/salary_synthetic_full_new.csv")
    adult_cleaned_syn = pd.read_csv("synthetic_datasets/adult_synthetic_ucd.csv")

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

    salary_cleaned_meta = SingleTableMetadata()
    salary_cleaned_meta.detect_from_dataframe(data=salary_cleaned)

    adult_cleaned_meta = SingleTableMetadata()
    adult_cleaned_meta.detect_from_dataframe(adult_cleaned_syn)

    print("Iris report:")
    iris_quality_report = evaluate_quality(iris_data, iris_syn, iris_meta)
    iris_column_shapes = iris_quality_report.get_details(property_name='Column Shapes')\
        .sort_values(by=["Quality Score"], ascending=False)
    iris_column_shapes_latex = iris_column_shapes.to_latex(index=False)

    print("Mushroom report:")
    mushroom_quality_report = evaluate_quality(mushroom_data, mushroom_syn, mushroom_meta)
    mushroom_column_shapes = mushroom_quality_report.get_details(property_name='Column Shapes')\
        .sort_values(by=["Quality Score"], ascending=False)
    mushroom_column_shapes_latex = mushroom_column_shapes.to_latex(index=False)

    print("Salary report:")
    salary_quality_report = evaluate_quality(salary_data, salary_syn, salary_meta)
    salary_column_shapes = salary_quality_report.get_details(property_name='Column Shapes')\
        .sort_values(by=["Quality Score"], ascending=False)
    salary_column_shapes_latex = salary_column_shapes.to_latex(index=False)

    print("Salary cleaned report:")
    salary_cleaned_quality_report = evaluate_quality(salary_cleaned, salary_cleaned_syn, salary_cleaned_meta)
    salary_cleaned_column_shapes = salary_cleaned_quality_report.get_details(property_name='Column Shapes') \
        .sort_values(by=["Quality Score"], ascending=False)
    salary_cleaned_column_shapes_latex = salary_cleaned_column_shapes.to_latex(index=False)

    print("Adult report:")
    adult_quality_report = evaluate_quality(adult_data, adult_syn, adult_meta)
    adult_column_shapes = adult_quality_report.get_details(property_name='Column Shapes')\
        .sort_values(by=["Quality Score"], ascending=False)
    adult_column_shapes_latex = adult_column_shapes.to_latex(index=False)

    print("Adult cleaned report:")
    adult_cleaned_quality_report = evaluate_quality(adult_cleaned, adult_cleaned_syn, adult_cleaned_meta)
    adult_cleaned_column_shapes = adult_cleaned_quality_report.get_details(property_name='Column Shapes') \
        .sort_values(by=["Quality Score"], ascending=False)
    adult_cleaned_column_shapes_latex = adult_cleaned_column_shapes.to_latex(index=False)


