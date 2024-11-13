from google.cloud import aiplatform

aiplatform.init(project="your_project_id", location="us-central1")

job = aiplatform.CustomTrainingJob(
    display_name="gpu_training_job",
    container_uri="gcr.io/cloud-aiplatform/training/tf-gpu.2-4:latest",
    # Specify machine type and GPUs
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)

# Run the job
job.run(
    args=["--epochs=10", "--batch_size=32"],
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",
    accelerator_count=1,
)
