
# [Team AutoML Grandmasters] Fourth AutoML Grand Prix Competition Write-Up


See [here](https://www.kaggle.com/competitions/playground-series-s4e8/discussion/523656) for the detailed post competition write-up.

This repository additionally provides the used the code to run AutoGluon with the same settings as we did (`main.py`), the required requirements (`requirements.txt`), and
an additional overview picture (`overview.jpg`).

Furthermore, `autogluon_distributed_example.py` contains an example on how to use the prototype of AutoGluon distributed (see [here](https://github.com/LennartPurucker/autogluon/tree/distributed_autogluon) for code). 
Likewise `ag_post_hoc_ensembler.py` contains code for building a post hoc ensemble of an AutoGluon run and additional predictions or prediction probabilities. 


## Reproducing Our Submission
<!---
A link to a code repository with complete and detailed instructions so that the results obtained can be reproduced. This is recommended for all participants and mandatory for winners to claim the prizes.
-->

To reproduce (not replicate as this is essentially impossible given different hardware and time-based randomness) our submission, follow these steps:
* Install the required python packages specified in `requirements.txt`. 
* Download the test and train data from Kaggle.
* Edit the paths to the data in the `main.py` file.
* Run the `main.py` file.

To apply our Kaggle tricks, see `ag_post_hoc_ensembler.py` and adjust the input data and paths accordingly.
We build an ensemble using all models of a default one-hour run of AutoGluon (with 192 CPUs) run and the final predictions from a four-hour run of AutoGluon (with 1000 CPUs).

## Main Contributions List
<!--- An itemized list of your main contributions and critical elements of success. Suggestions: Contrast your proposed method with others e.g. in terms of computational or implementation complexity, parallelism, memory cost, theoretical grounding.
-->
* Remove categories that do not exist in test data from the train data.
* Use log loss and early stopping metric. 
* Use 16-fold cross-validation with 1-layer stacking.
* Use a custom portfolio of models, meta-learned from TabRepo and 100 instead of 25 iterations for post hoc ensembling.
* Building a weighted ensemble of the four-hour run and all models of the one-hour run while increasing the decimal precision of the weighted ensemble.
* Use a distributed version of AutoGluon to fit with 1000 CPUs. 


## Detailed Methodology
<!--- A detailed description of methodology. Expand and explain your contributions in more detail. The explanations must be self-contained and one must be able to reproduce the approach by reading this section. You can explain and justify the approach by any means, e.g. citations, equations, tables, algorithms, platforms and code libraries utilized, etc. A detailed explanation of the architecture, preprocessing, loss function, training details, hyper-parameters, etc. is expected.
-->
See [here](https://www.kaggle.com/competitions/playground-series-s4e8/discussion/523656) for more details.

## Workflow Diagram
<!--- A representative image / workflow diagram of the method to support additional description. -->
![AutoGluonOverview](overview.jpg)
