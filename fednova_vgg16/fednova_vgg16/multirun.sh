#!/bin/bash
 cd ..
# Array to store failed experiments
failed_experiments=()

# Function to run an experiment and track failures
run_experiment() {
  local dataset=$1
  local partitioner=$2
  local partition_param=$3
  local partition_value=$4
  local num_classes=$5

  # Generate timestamp for directory
  timestamp=$(date +%Y-%m-%d_%H-%M-%S)

  # Define output directory path with timestamp
  output_dir="outputs/${dataset}/${partitioner}/${partition_value}/${timestamp}"

  echo "Starting experiment with dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"

  # Run the experiment
  python -m fednova_vgg16.main hydra.run.dir=$output_dir dataset.subset="$dataset" dataset.partitioner.name="$partitioner" dataset.partitioner.$partition_param="$partition_value" model.num_classes="$num_classes"

  # Check if the experiment failed
  if [ $? -ne 0 ]; then
    echo "Experiment failed: dataset=${dataset}, partitioner=${partitioner}, ${partition_param}=${partition_value}, model.num_classes=${num_classes}"
    # Record the failed experiment
    failed_experiments+=("${dataset}/${partitioner}/${partition_value}")
  fi
}


## Run experiments for pathmnist with PathologicalPartitioner
#run_experiment "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 7 9
#run_experiment "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 4 9
#run_experiment "pathmnist" "PathologicalPartitioner" "num_classes_per_partition" 2 9

## Run experiments for pathmnist with DirichletPartitioner
#run_experiment "pathmnist" "DirichletPartitioner" "alpha" 0.9 9
run_experiment "pathmnist" "DirichletPartitioner" "alpha" 0.3 9
#run_experiment "pathmnist" "DirichletPartitioner" "alpha" 0.1 9
#
## Run experiments for tissuemnist with PathologicalPartitioner
#run_experiment "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 7 8
#run_experiment "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 4 8
#run_experiment "tissuemnist" "PathologicalPartitioner" "num_classes_per_partition" 2 8
#
## Run experiments for tissuemnist with DirichletPartitioner
#run_experiment "tissuemnist" "DirichletPartitioner" "alpha" 0.9 8
#run_experiment "tissuemnist" "DirichletPartitioner" "alpha" 0.3 8
#run_experiment "tissuemnist" "DirichletPartitioner" "alpha" 0.1 8

## Run experiments for bloodmnist with PathologicalPartitioner
#run_experiment "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 2 8
#run_experiment "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 4 8
#run_experiment "bloodmnist" "PathologicalPartitioner" "num_classes_per_partition" 7 8
#
## Run experiments for bloodmnist with DirichletPartitioner
#run_experiment "bloodmnist" "DirichletPartitioner" "alpha" 0.9 8
#run_experiment "bloodmnist" "DirichletPartitioner" "alpha" 0.3 8
#run_experiment "bloodmnist" "DirichletPartitioner" "alpha" 0.1 8

# Final report
echo "Experiments completed!"

# Check if there were any failures
if [ ${#failed_experiments[@]} -ne 0 ]; then
  echo "The following experiments failed:"
  for experiment in "${failed_experiments[@]}"; do
    echo "  - $experiment"
  done
  exit 1  # Exit with error code if any experiments failed
else
  echo "All experiments succeeded!"
  exit 0  # Exit with success code if everything went fine
fi
