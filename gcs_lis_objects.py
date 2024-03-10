import datetime
import asyncio
import aiohttp
from google.cloud import storage

async def list_objects_with_prefix(bucket_name, prefix):
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)

    return [blob.name async for blob in blobs]

async def check_recent_objects(bucket_name, prefix):
    current_time = datetime.datetime.utcnow()
    thirty_days_ago = current_time - datetime.timedelta(days=30)

    recent_objects = []
    async with aiohttp.ClientSession() as session:
        blobs = await list_objects_with_prefix(bucket_name, prefix)
        tasks = [check_object_modification_time(bucket_name, blob_name, thirty_days_ago, session) for blob_name in blobs]
        recent_objects = await asyncio.gather(*tasks)

    return [blob_name for blob_name in recent_objects if blob_name is not None]

async def check_object_modification_time(bucket_name, blob_name, threshold_time, session):
    url = f"https://storage.googleapis.com/storage/v1/b/{bucket_name}/o/{blob_name}"
    async with session.get(url) as response:
        if response.status == 200:
            data = await response.json()
            update_time_str = data.get("updated", None)
            if update_time_str:
                update_time = datetime.datetime.strptime(update_time_str[:-1], "%Y-%m-%dT%H:%M:%S")
                if update_time >= threshold_time:
                    return blob_name
    return None

if __name__ == "__main__":
    bucket_name = "your_bucket_name"
    folder_prefix = "your/folder/prefix/"  # Include the trailing slash for a folder

    loop = asyncio.get_event_loop()
    recent_objects = loop.run_until_complete(check_recent_objects(bucket_name, folder_prefix))

    if recent_objects:
        print("Recently modified or created objects in the folder:")
        for obj in recent_objects:
            print(obj)
    else:
        print("No recently modified or created objects found in the folder.")
