from google.cloud import dataproc_v1
from google.protobuf.json_format import MessageToDict
import yaml

def get_dataproc_cluster_yaml(project_id, region, cluster_name, output_file):
    try:
        # Initialize the Dataproc client
        client = dataproc_v1.ClusterControllerClient(
            client_options={"api_endpoint": f"{region}-dataproc.googleapis.com:443"}
        )
        
        # Get the cluster configuration
        cluster = client.get_cluster(project_id=project_id, region=region, cluster_name=cluster_name)
        
        # Convert the cluster configuration to a dictionary
        cluster_dict = MessageToDict(cluster._pb)

        # Convert the dictionary to YAML
        cluster_yaml = yaml.dump(cluster_dict, default_flow_style=False)

        # Write the YAML to the output file
        with open(output_file, 'w') as f:
            f.write(cluster_yaml)

        print(f"Cluster configuration exported to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Replace with your project ID, region, cluster name, and desired output file path
project_id = 'your-project-id'
region = 'your-region'
cluster_name = 'your-cluster-name'
output_file = 'cluster_config.yaml'

get_dataproc_cluster_yaml(project_id, region, cluster_name, output_file)
