# Hand Wash classification using Asus ZenWatch 2 data

## Resources
- Asus ZenWatch 2
- Wada.jar for extracting data from watch
- WeKA for performing classification and statistical analysis

## Instructions

In order to run the files, create a virtual environment using conda and run the following:

```
pip install -r requirements.txt
```

The steps for the scripts are as follows:

- Add the csv files in the <mark>raw_data</mark> folder
- Run the the <mark>convert_csv_to_arff.py</mark> script as follows:
  
  ```
  python convert_csv_to_arff.py --csv_path <path_to_the_csv> --arff_path <path_to_the_arff> 
  ```

  path_to_the_csv: It is the path to the csv file which needs to be read and converted to arff format
  path_to_the_arff: It is the path where the arff file would be saved

The script will generate the arff file in the folder <mark>arff_data</mark> folder as well as output the accuracy of the decision tree.