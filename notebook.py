from google.cloud import aiplatform_v1

# Replace these variables with your values
project_id = 'your-project-id'
location = 'us-central1'  # or your preferred location
instance_name = 'your-notebook-instance-name'
machine_type = 'n1-standard-4'  # or your preferred machine type
accelerator_type = 'NVIDIA_TESLA_V100'
accelerator_count = 1

# Initialize the AI Platform (Unified) client
client = aiplatform_v1.NotebookServiceClient()

# Construct the instance resource
instance = {
    'name': instance_name,
    'location': location,
    'type': 'INSTANCE',
    'notebook_instance': {
        'instance_type': machine_type,
        'accelerator_config': {
            'type': accelerator_type,
            'core_count': accelerator_count
        }
    }
}

# Create the notebook instance
parent = f"projects/{project_id}/locations/{location}"
response = client.create_notebook_instance(parent=parent, notebook_instance=instance)

print("Created notebook instance:", response.name)
