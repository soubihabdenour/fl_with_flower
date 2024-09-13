export CUDA_VISIBLE_DEVICES=1
cd ..
# Experiment 1
echo "Starting Experiment 1 "

python -m fedavg_mobilnet_poisoned.main dataset.subset="tissuemnist" dataset.partitioner.name="DirichletPartitioner" dataset.partitioner.alpha=0.1 model.num_classes=8

# Check if Experiment 1 succeeded
if [ $? -ne 0 ]; then
  echo "Experiment 1 failed!"
  exit 1
fi

echo "Experiments completed successfully!"