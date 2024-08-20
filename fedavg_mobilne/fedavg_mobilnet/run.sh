
cd ..
#!/bin/bash

# Experiment 1
echo "Starting Experiment "

python -m fedavg_mobilnet.main num_rounds=100

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

echo "Starting Experiment 2"
python -m fedavg_mobilnet.main num_rounds=100 dataset.partitioner.name="DirichletPartitioner"
# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

# Experiment 2
echo "Starting Experiment 2"

python -m fedavg_mobilnet.main num_rounds=100 dataset.partitioner.name="PathologicalPartitioner"

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 2 failed!"
  exit 1
fi

echo "Starting Experiment 1"
python -m fedavg_mobilnet.main num_rounds=100 dataset.partitioner.name="DirichletPartitioner"
# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

echo "Experiments completed successfully!"