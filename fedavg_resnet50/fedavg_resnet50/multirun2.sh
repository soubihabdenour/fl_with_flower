#!/bin/bash

# Function to run an experiment and check if it failed
run_experiment() {
  "$@"
  if [ $? -ne 0 ]; then
    echo "$1 failed!"
  else
    echo "$1 succeeded!"
  fi
}

# Experiment 3
echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet5.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=9

# Experiment 2
echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=9

# Experiment 1
echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=9

# Experiment 3 (Second Set)
echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=9

# Experiment 2 (Second Set)
echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=9

# Experiment 1 (Second Set)
echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=9

# BloodMNIST Experiments
echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=8

echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=8

echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=8

echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=8

echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=8

echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

# TissueMNIST Experiments
echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=8

echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=8

echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=8

echo "Starting Experiment 3"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=8

echo "Starting Experiment 2"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=8

echo "Starting Experiment 1"
run_experiment python -m fedavg_resnet50.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

echo "Experiments completed!"
