# AI Safety is Safe AI: An Empirical Study
This is a brief readme to describe how to reproduce the results of the titular paper under review at Nature Machine Intelligence, and the data supplied for the reproduction.

## Instructions for reproduction

To reproduce our figures in the paper you will need Python 3.11 installed on your computer. 

Run the following commands from the folder in which `supplementary.zip` is located:

```bash
unzip supplementary.zip
pip install -r requirements.txt
python code/analysis.py
```

This will recreate all figures in the paper into a folder called `output`.

## Data
The supplementary material also contains two data folders.

`data/export` contains all retrieved non-duplicate publications in a standard JSON format.

`data/annotations` contains all selected publications with annotations as described in Section 2.2 in a JSON format.