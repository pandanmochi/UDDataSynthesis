import datasyn as ds
import json
import pandas as pd

if __name__ == '__main__':
    # Open the JSON file
    with open('glassdoor_salaries.json') as file:
        json_data = json.load(file)

    # Convert JSON data to a DataFrame
    data_3 = pd.DataFrame(json_data)

    data_4 = data_3.drop(columns=['Dept', 'Education'])
    cdg = ds.get_cdg(data_3)
    print(f"cdg: {cdg}")

    new_data = ds.generate_data(cdg, 1000)

