# Project: Notebook to Python Conversion

## Task
- Convert a set of .ipynb notebooks to .py scripts
- The scripts should be executable from terminal
- The source notebooks are listed below

## Source Notebooks
Location: /Users/adityaponnada/Documents/codework/real_time_prompting/real_time_prompting/
Only convert the .ipynb files listed below
If there is a .py file listed, use it as is
Conversion order: import_prep_dataset.ipynb, compute_raw_features.py, feature_selection_normalization.ipynb, held_out_data_prep.ipynb, general_rnn.ipynb, hybrid_rnn.ipynb, prep_withdrawn_data.ipynb, withdrew_general_eval.ipynb, withdrew_hybrid_eval.ipynb, survival_analysis.ipynb

## Target Structure to Create
Create the following structure:
project/
├── src/
│   ├── __init__.py
│   └── [module files derived from notebooks]
├── cli/
│   └── main.py          # entry point, argparse-based CLI
├── config/
│   └── config.yaml      # any hardcoded values moved here
├── tests/
│   └── test_*.py
├── requirements.txt
├── README.md
└── CLAUDE.md
└── .gitignore

## Conversion Rules
- Consolidate all import statements to the top of the script
- Each notebook becomes one or more modules in src/
- Strip out exploratory/scratch cells (cells with just print statements
  or intermediate checks) unless they serve a clear purpose
- Convert notebook parameters/constants into config.yaml entries
- All functions must have type hints and docstrings
- The CLI in cli/main.py should expose the main pipeline as a command
- Keep the intermediate file creation as is
- Use a common folder based on user input to save the intermediate files
- The output folder location should be declared only once in the repo
- README should have information on creating a virtual environment to run tensorflow (python 3.11.14)
- If there are parts of the code duplicated across notebooks, include a helper.py script that can contain all the common helper functions
- Assess the functions carefully and add a helper if the function implementation is exactly the same across notebooks
- All the function print output should remain as is and should print to terminal
- Remove any redundant print statements
- The .h5 model files can be saved inside the repo
- All the figures should be saved in the single output folder that user inputs
- Figure file names should be self-explanatory
- All the evaluation and statistical results such as t-tests, log rank test for survival analysis, model metrics like accuracy, f1 tables, optimum model thresholds, gtcn model summary should be saved as separate .txt files in the putput folder (named appropriately)