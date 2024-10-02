#!/bin/bash
cd ..
# List to store failed experiments
FAILED_EXPERIMENTS=()

# Function to run an experiment and check if it failed
run_experiment() {
  EXPERIMENT_NAME=$1
  shift
  echo "Running $EXPERIMENT_NAME"
  "$@"
  
  if [ $? -ne 0 ]; then
    echo "$EXPERIMENT_NAME failed!"
    FAILED_EXPERIMENTS+=("$EXPERIMENT_NAME")
  else
    echo "$EXPERIMENT_NAME succeeded!"
  fi
}

# PathMNIST Experiments
run_experiment "PathMNIST: PathologicalPartitioner, num_classes_per_partition=7" \
  python -m df.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=9

run_experiment "PathMNIST: PathologicalPartitioner, num_classes_per_partition=4" \
  python -m fedavg_resnet50.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=9

run_experiment "PathMNIST: PathologicalPartitioner, num_classes_per_partition=2" \
  python -m df.main dataset.subset="pathmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=9

run_experiment "PathMNIST: DirichletPartitioner, alpha=0.9" \
  python -m df.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=9

run_experiment "PathMNIST: DirichletPartitioner, alpha=0.3" \
  python -m df.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=9

run_experiment "PathMNIST: DirichletPartitioner, alpha=0.1" \
  python -m df.main dataset.subset="pathmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=9

# BloodMNIST Experiments
run_experiment "BloodMNIST: PathologicalPartitioner, num_classes_per_partition=7" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=8

run_experiment "BloodMNIST: PathologicalPartitioner, num_classes_per_partition=4" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=8

run_experiment "BloodMNIST: PathologicalPartitioner, num_classes_per_partition=2" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=8

run_experiment "BloodMNIST: DirichletPartitioner, alpha=0.9" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=8

run_experiment "BloodMNIST: DirichletPartitioner, alpha=0.3" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=8

run_experiment "BloodMNIST: DirichletPartitioner, alpha=0.1" \
  python -m df.main dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

# TissueMNIST Experiments
run_experiment "TissueMNIST: PathologicalPartitioner, num_classes_per_partition=7" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=8

run_experiment "TissueMNIST: PathologicalPartitioner, num_classes_per_partition=4" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=8

run_experiment "TissueMNIST: PathologicalPartitioner, num_classes_per_partition=2" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=8

run_experiment "TissueMNIST: DirichletPartitioner, alpha=0.9" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=8

run_experiment "TissueMNIST: DirichletPartitioner, alpha=0.3" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=8

run_experiment "TissueMNIST: DirichletPartitioner, alpha=0.1" \
  python -m df.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

# After all experiments are done, print the failed ones
if [ ${#FAILED_EXPERIMENTS[@]} -ne 0 ]; then
  echo "The following experiments failed:"
  for experiment in "${FAILED_EXPERIMENTS[@]}"; do
    echo "$experiment"
  done
else
  echo "All experiments completed successfully!"
fi
