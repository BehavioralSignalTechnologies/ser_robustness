# 🚀 Code Structure

- `src/parsing`: contains the code for parsing the datasets.
- `src/evaluation`: contains the code for evaluating the model predictions.
- `src/noise_generation`: contains the scripts for adding noise to the datasets.

# 📦 Adding support for a new dataset 

If you want to add support for a new dataset, you have to follow these steps:

## Step 1: Implement the parser class for the dataset

The parser should be placed in the `src/parsing` directory and should inherit from the `Parser` class.
It should implement the method `run_parser` which should return a dictionary with the following structure:
```
{
    '/path/to/audio_file_1.wav': {'emotion': 'class_name', ...},
    ...
}
```

You can refer to the `ParserForIEMOCAP` class for an example of how to implement a parser.

## Step 2: Add the parser to the `get_parser` function

In the `src/parsing/get_parser.py` file, add an entry for the new parser in the `get_parser` function.
At this point, you should be able to use the `corrupt_dataset.py` script to add noise to the new dataset.
Because this script copies the dataset in the exact same structure as the original dataset, your code should work out of the box.

## (Optional) Step 3: Add evaluation support

If you want to add support for evaluating the model predictions on the new dataset, you should update the `src/evaluation/evaluate.py` script.

## Step 4: Open a pull request

We encourage you to open a pull request with the new dataset support as this will help the community.
Please update the README with the new dataset information and add a link to the dataset.
