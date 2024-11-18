import unittest
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2 import service_account

# Configuration
PROJECT_ID = "your-project-id"
SERVICE_ACCOUNT_EMAIL = "your-service-account@your-project-id.iam.gserviceaccount.com"
KEY_RING_LOCATION = "us-central1"  # Replace with your key ring location
KEY_RING_NAME = "your-key-ring-name"
KEY_NAME = "your-key-name"
CMEK_RESOURCE = f"projects/{PROJECT_ID}/locations/{KEY_RING_LOCATION}/keyRings/{KEY_RING_NAME}/cryptoKeys/{KEY_NAME}"

# Permissions to check
REQUIRED_PERMISSIONS = {
    "cloud_functions": [
        "cloudfunctions.functions.create",
        "cloudfunctions.functions.update",
        "cloudfunctions.functions.get",
    ],
    "cloud_storage": [
        "storage.buckets.get",
        "storage.objects.create",
        "storage.objects.get",
    ],
    "iam": ["iam.serviceAccounts.actAs"],
    "cloud_kms": [
        "cloudkms.cryptoKeyVersions.useToEncrypt",
        "cloudkms.cryptoKeyVersions.useToDecrypt",
        "cloudkms.cryptoKeys.get",
        "cloudkms.cryptoKeyVersions.get",
    ],
    "logging": ["logging.logEntries.create", "logging.logEntries.list"],
    "monitoring": ["monitoring.metricDescriptors.get"],
}

# Helper function to check permissions
def check_permissions(credentials, resource, permissions):
    service = build("cloudresourcemanager", "v1", credentials=credentials)
    request = service.projects().testIamPermissions(resource=resource, body={"permissions": permissions})
    response = request.execute()
    granted_permissions = response.get("permissions", [])
    missing_permissions = set(permissions) - set(granted_permissions)
    return granted_permissions, missing_permissions

class TestServiceAccountPermissions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.credentials = service_account.Credentials.from_service_account_file(
            "path/to/service-account-key.json",
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )
    
    def test_cloud_functions_permissions(self):
        granted, missing = check_permissions(
            self.credentials, f"projects/{PROJECT_ID}", REQUIRED_PERMISSIONS["cloud_functions"]
        )
        self.assertFalse(missing, f"Missing Cloud Functions permissions: {missing}")

    def test_cloud_storage_permissions(self):
        granted, missing = check_permissions(
            self.credentials, f"projects/{PROJECT_ID}", REQUIRED_PERMISSIONS["cloud_storage"]
        )
        self.assertFalse(missing, f"Missing Cloud Storage permissions: {missing}")

    def test_iam_permissions(self):
        granted, missing = check_permissions(
            self.credentials, f"projects/{PROJECT_ID}", REQUIRED_PERMISSIONS["iam"]
        )
        self.assertFalse(missing, f"Missing IAM permissions: {missing}")

    def test_cloud_kms_permissions(self):
        granted, missing = check_permissions(
            self.credentials, CMEK_RESOURCE, REQUIRED_PERMISSIONS["cloud_kms"]
        )
        self.assertFalse(missing, f"Missing Cloud KMS permissions: {missing}")

    def test_logging_permissions(self):
        granted, missing = check_permissions(
            self.credentials, f"projects/{PROJECT_ID}", REQUIRED_PERMISSIONS["logging"]
        )
        self.assertFalse(missing, f"Missing Logging permissions: {missing}")

    def test_monitoring_permissions(self):
        granted, missing = check_permissions(
            self.credentials, f"projects/{PROJECT_ID}", REQUIRED_PERMISSIONS["monitoring"]
        )
        self.assertFalse(missing, f"Missing Monitoring permissions: {missing}")

if __name__ == "__main__":
    unittest.main()
