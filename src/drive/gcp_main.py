#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 13:19:40 2018

@author: batman
"""
#%% Setup
import os
import google.cloud.storage

from pprint import pprint

#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/batman/gcloud_credentials/Test First-48dd42d10d05.json"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/batman/gcloud_credentials/MJAccessTest Storage.json"

#%% Client
# Create a storage client.
storage_client = google.cloud.storage.Client()
print(storage_client)

#%% Create
# The name for the new bucket
bucket_name = 'my-new-bucket88776'

# Creates the new bucket
bucket = storage_client.create_bucket(bucket_name)

print('Bucket {} created.'.format(bucket.name))

bucket_name = 'my-new-bucket88776'
bucket = storage_client.get_bucket(bucket_name)
l = dir(bucket)
pprint(l)
#%% Upload
# Select a file
path_file = r'/media/batman/USB STICK/catdog6/run003/weights-epoch30-0.86.hdf5'
assert os.path.exists(path_file)
blob_name = os.path.basename(path_file)
blob = bucket.blob(blob_name)

# Upload the local file to Cloud Storage.
blob.upload_from_filename(path_file)

print('File {} uploaded to {}.'.format(
    source_file_name,
    bucket))

#%% Download

blobs = bucket.list_blobs()
for blob in blobs:
    print(blob.name)


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))




