# Hand Wash classification using Asus ZenWatch 2 data

## Resources
- Asus ZenWatch 2
- Wada.jar for extracting data from watch
- WeKA for performing classification and statistical analysis

## Instructions

In order to run the files, create a virtual environment using conda and run the following:

- Install libomp in your OS

```
pip install -r requirements.txt
```

The steps for the scripts are as follows:

- Add the csv files in the <mark>indoor_walk</mark> and <mark>outdoor_walk</mark> folder
- Run the the <mark>model_eval.py</mark> script as follows:
  
  ```
  python model_eval.py --add_extra_cols --classifier <classifier_option>
  ```
  classifier_option: In order to run respective classifier provide the codes accordingly
  - 1 -> Random Forrest
  - 2 -> XGBoost

The script will generate <mark>results_extra_cols_{..}_{classifier_name}.csv</mark> file.