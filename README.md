# software-framework

AK4 software framework 

### install

```conda create -n qcframework python==3.11```

for each use case do:

```cd {use case folder}```

and then do

```pip install -r requirements.txt```

use

```cd ../```

to go back into the root folder.

### Execute examples

```python examples/pas_example_cplex.py```

### create your own code

```python installer.py``

Select the use cases you want to use.
Click install selected use cases.

Then you can write:

```
from framework_config import model_classes
```

and use these clases to create your own examples

## License

This project is licensed under the [Apache License 2.0](LICENSE.txt). In order to use parts of the code presented in a non-production environment, additional [Gurobi](https://pypi.org/project/gurobipy/) and [CPLEX](https://pypi.org/project/cplex/) licenses are required. Please read the license statements carefully before using or distributing the code.

## Funding
<a href="https://www.bmwk.de/"><img src="logoBMWK.svg" height="150px" /></a>

