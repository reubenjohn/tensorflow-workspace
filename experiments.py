from datetime import datetime

LOG_ROOT_DIR = '/scratch/logs'

DATASET_ROOT_DIR = '/scratch/datasets'

def form_log_directory_path(experiment_name: str):
    return '%s/eye_of_newt/%s/%s' % (LOG_ROOT_DIR, experiment_name, datetime.now().isoformat())
