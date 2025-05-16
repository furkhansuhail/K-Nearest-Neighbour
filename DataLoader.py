from dataclasses import dataclass
from pathlib import Path
import urllib.request as request


Dataset_Link = "https://github.com/furkhansuhail/ProjectData/raw/refs/heads/main/K%20Nearest%20Neighbour/Breast_cancer_data.csv"
# Step 1: Configuration class for downloading data
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    STATUS_FILE: str
    ALL_REQUIRED_FILES: list


# Step 2: Config object
config = DataIngestionConfig(
    root_dir=Path("Dataset"),
    source_URL=Dataset_Link,
    local_data_file=Path("Dataset/Breast_cancer_data.csv"),
    STATUS_FILE="Dataset/status.txt",
    ALL_REQUIRED_FILES=[]
)


def download_project_file(source_URL, local_data_file):
    local_data_file.parent.mkdir(parents=True, exist_ok=True)
    if local_data_file.exists():
        print(f"✅ File already exists at: {local_data_file}")
    else:
        print(f"⬇ Downloading file from {source_URL}...")
        file_path, _ = request.urlretrieve(url=source_URL, filename=local_data_file)
        print(f"✅ File downloaded and saved to: {file_path}")

download_project_file(config.source_URL, config.local_data_file)