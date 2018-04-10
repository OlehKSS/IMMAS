# Intelligent Mammogram Mass Analysis and Segmentation (IMMAS)
Repository for Advanced Image Analysis project.


## Git guidelines

 Please make changes in your local branch and then open a pull request to the **master** branch. For branch naming please use patterns `github_user_name/IMMAS-issue_number` or just `IMMAS-issue_number`. When merging your pull request do not forget to add corresponding issue number in commit message, e. g. "Fixes issue #11", so that issue will be automatically closed.


 ## Installing and testing
Create a virtual environment with the name `immas`
```bash
conda create --name immas python=3
```

Activate the created environment via 
```bash
source activate immas
```
Then run ``python setup.py develop`` for development purposes or ``python setup.py install`` if you want to do installation.

The environment can be deactivated with the following command
```bash
source deactivate immas
```

If you want to test IMMAS with Jupyter-Notebook from your virtual environment you should add new kernel using these commands
```bash
source activate immas

conda install ipykernel

python -m ipykernel install --user --name immas --display-name "Python 3 (IMMAS)"
```
and select this kernel while running Jupyter-Notebook.