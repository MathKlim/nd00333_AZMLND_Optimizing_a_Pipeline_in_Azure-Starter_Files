# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains marketing data about individuals. The data is related with direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict whether the client will subscribe a bank term deposit (column y).

The best performing HyperDrive model was model with ID `HD_bf0c2c3d-713b-44ce-ac46-80f90e50feb0_9`. It derived from a Scikit-learn pipeline and had an accuracy of **0.914** with run parameters `['--C', '0.20624530875187946', '--max_iter', '125']`.

In contrast, for the AutoML the best model was model with ID `AutoML_b07a763b-8e10-44da-925e-b794e0fe356c_27`, the accuracy was **0.918** and the algorithm used was VotingEnsemble.

Although the best model comes from AutoML, the difference with hyperparameters tuning is only up to $10^{-3}$

## Scikit-learn Pipeline
I specified the parameter sampler as such:

```python
ps = RandomParameterSampling( {
        "--C": uniform(0.1, 1.0),
        "--max_iter": choice(25,50,75,100,125,150,175,200)
    }
)
```

I chose a random uniform sampling for `C` discrete values with choice for `max_iter`.

`C` is the Regularization while `max_iter` is the maximum number of iterations.

`RandomParameterSampling` is one of the choices available for the sampler and I chose it because it is the faster and supports early termination of low-performance runs. We could use `GridParameterSampling` to exhaustively search over the search space or `BayesianParameterSampling` to explore the hyperparameter space.

An early stopping policy is used to automatically terminate poorly performing runs thus improving computational efficiency. I chose the BanditPolicy which I specified as follows:

`policy = BanditPolicy(evaluation_interval=1, slack_factor=0.2, slack_amount=None, delay_evaluation=0)`

evaluation_interval: This is optional and represents the frequency for applying the policy. Each time the training script logs the primary metric counts as one interval.

slack_factor: The amount of slack allowed with respect to the best performing training run. This factor specifies the slack as a ratio.

Any run that doesn't fall within the slack factor or slack amount of the evaluation metric with respect to the best performing run will be terminated.

Full configuration.


```python
# Specify a Policy
policy = BanditPolicy(evaluation_interval=1, slack_factor=0.2, slack_amount=None, delay_evaluation=0)

if "training" not in os.listdir():
    os.mkdir("./training")

# Setup environment for your training run
# Note that conda isn't the only spec available, you can set up compute env with pip or a dockerfile.
sklearn_env = Environment.from_conda_specification(name='sklearn-env', file_path='conda_dependencies.yml')

# Create a ScriptRunConfig Object to specify the configuration details of your training job
src = ScriptRunConfig(source_directory='.',
                            script='train.py',
                            compute_target=cpu_cluster,
                            environment=sklearn_env)

# Create a HyperDriveConfig using the src object, hyperparameter sampler, and policy.
hyperdrive_config = HyperDriveConfig(run_config=src,
                                hyperparameter_sampling=ps,
                                policy=policy,
                                primary_metric_name='Accuracy',
                                primary_metric_goal=PrimaryMetricGoal.MAXIMIZE,
                                max_total_runs=10,
                                max_concurrent_runs=4)
```

## AutoML

The configuration for the AutoML run is the following:

```python
automl_config = AutoMLConfig(
                             experiment_timeout_minutes=30,
                             task="classification",
                             training_data = ds,
                             label_column_name = "y",
                             compute_target=cpu_cluster,
                             iterations=30,
                             iteration_timeout_minutes=5,
                             primary_metric="accuracy",
                             n_cross_validations=5
                            )
```

* task='classification' defines the experiment type which in this case is classification.
* primary_metric='accuracy', accuracy as the primary metric, which is the classical metric to use for a classfication task, although not the only one (AUC, precision, recall, F1-score, etc.).
* n_cross_validations=5 sets how many cross validations to perform, based on the same number of folds (number of subsets), we chose 5 folds for cross-validation. The metrics are calculated with the average of the 5 validation metrics.


***
## Pipeline comparison
**Comparison of the two models and their performance. Differences in accuracy & architecture - comments**


| HyperDrive Model |                                           |
| :--------------: | :---------------------------------------: |
|        id        | HD_bf0c2c3d-713b-44ce-ac46-80f90e50feb0_9 |
|     Accuracy     |                   0.914                   |


| AutoML Model |                                                |
| :----------: | :--------------------------------------------: |
|      id      | AutoML_b07a763b-8e10-44da-925e-b794e0fe356c_27 |
|   Accuracy   |               0.9180880121396056               |
|  Algortithm  |                 VotingEnsemble                 |


The difference in accuracy between the two models is not breathtaking and although the AutoML model performed better in terms of accuracy, it's not cristal clear from these metrics we needed it ar first. AutoML seems to be a good choice when you want to set up a good **baseline** to beat with more advanced methods or data engineering.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**


Our data is **imbalanced**.

Class imbalance is a very common issue in classification problems in machine learning.

To deal with imbalanced datas we can use:
1. A different metric; for example, AUC_weighted which is more fit for imbalanced data
2. Generate synthetic datas from the imbalanced classe, with algortihm like [SMOTE](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html), or even neural generative neural networks like Variational Auto-Encoder or Generative Adversarial Networks.
3. Using neural networks, we could use the [Focal loss](https://arxiv.org/abs/1708.02002), which is a loss specialized for imbalanced datasets.

