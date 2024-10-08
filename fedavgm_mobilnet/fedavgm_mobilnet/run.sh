
cd ..
#!/bin/bash

# Experiment 2
echo "Starting Experiment 3"

python3 -m fedavgm_mobilnet.main dataset.partitioner.name="PathologicalPartitioner"

# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 3 failed!"
  exit 1
fi

echo "Starting Experiment 2"
python3 -m fedavgm_mobilnet.main dataset.partitioner.name="DirichletPartitioner"
# Check if Experiment 2 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 2 failed!"
  exit 1
fi
# Experiment 1
echo "Starting Experiment 1 "

python3 -m fedavgm_mobilnet.main

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

echo "Experiments completed successfully!"