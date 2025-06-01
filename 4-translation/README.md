# 4-translation

In this folder we perform the translation task, attempting the use of different models.

As mentioned in some other folder, 
the models Jupyter notebooks have corresponding Python script counterparts. This dual format serves a practical purpose: while notebooks provide an excellent interactive development environment with inline visualizations and step-by-step execution, they are less suitable for automated or long-running processes. The Python scripts enable seamless execution in virtual machines and remote servers, particularly for computationally intensive tasks that need to run unattended (such as overnight training runs or batch processing jobs). This approach allows us to leverage the exploratory advantages of notebooks during development while maintaining the operational flexibility needed for production-like environments.

Here are the details of the single files in this folder:
- `1-GPT2_finetuning.ipynb`: contains code for finetuning of GPT2 decoder model.
- `1b-GPT2_finetuning`: corresponding Python file.
- `2-BART_training.ipynb`: contains code for finetuning of BART encoder-decoder model.
- `2b-BART_training.py`: corresponding Python file.
- `3-testing.ipynb`: tests the various models on our test datasets and produce a table with results of some metrics.
- `4-test_results.ipynb`: summarizes the results obtained in the previous step by making several plots.
- `compute_metrics.py`: define the metrics we will use as compute_metrics parameter in our models.
- `load_dataset.py`: defines some function for dataset preparation useful to feed data in encoder-decoder models.