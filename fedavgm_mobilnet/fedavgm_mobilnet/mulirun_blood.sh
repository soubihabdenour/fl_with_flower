cd ..

# Experiment 2
echo "Starting Experiment 3"

python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=7 model.num_classes=8

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

echo "Starting Experiment 2"
python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=4 model.num_classes=8
# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 2 failed!"
  exit 1
fi
# Experiment 1
echo "Starting Experiment 1 "

python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="PathologicalPartitioner" dataset.partitioner.num_classes_per_partition=2 model.num_classes=8

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

# Experiment 2
echo "Starting Experiment 3"

python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.9 model.num_classes=8

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

echo "Starting Experiment 2"
python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.3 model.num_classes=8
# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 2 failed!"
  exit 1
fi
# Experiment 1
echo "Starting Experiment 1 "

python3 -m fedavgm_mobilnet.main num_rounds=100 dataset.subset="bloodmnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

echo "Experiments completed successfully!"