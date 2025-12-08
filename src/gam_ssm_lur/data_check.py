"""
I am going to check if the file is presesent. 
If missing, I will go ahead and download it from the URL.
"""

from pathlib import Path
import requests, zipfile, os, logging


def check_data_availability(data_path: Path, download_url: str, unzip_to: Path):
    """
    Check if the data file exists at the specified path.
    If not, download it from the given URL and extract it.
    """

    logger = logging.getLogger(__name__)

    if not data_path.exists():
        logger.warning(f"Data file {data_path} not found. Attempting to download...")

        zip_path = unzip_to / "data.zip"
        unzip_to.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists

        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status()  # Raise an error for bad responses

            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            logger.info(f"Data downloaded successfully to {zip_path}.")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(unzip_to)

            zip_path.unlink()  # Remove the zip file after extraction
            logger.info(f"Data extracted to {unzip_to}.")
            logger.info(
                f"Extractiopn completed. Data file is now available at {data_path}."
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download data from {download_url}: {e}")
            raise
        except zipfile.BadZipFile as e:
            logger.error(f"Failed to unzip the downloaded file: {e}")
            raise

    else:
        logging.info(f"Data file already exists at {data_path}.")