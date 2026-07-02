"""
Data fetching from Zenodo.

If the local ``data/`` directory is missing required files, resolves the
Zenodo concept DOI to whatever version is currently latest, downloads the
archived ``data.zip``, and extracts it in place.

Using the concept DOI (rather than a pinned version DOI) means this always
fetches the most recent dataset; see the Data Availability section of the
manuscript for the version DOI corresponding to the published results.
"""

from __future__ import annotations

import logging
import zipfile
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

ZENODO_CONCEPT_DOI = "10.5281/zenodo.16534137"

REQUIRED_FILES = (
    "features.csv",
    "target.csv",
    "grid/grid.geojson",
    "time_series/epa_timeseries.csv",
    "time_series/satellite_retreavals.csv",
    "time_series/traffic_timeseries.csv",
    "time_series/wind_sector_2023-06_daily.csv",
)


def _resolve_latest_record_id(concept_doi: str = ZENODO_CONCEPT_DOI) -> str:
    """Follow the concept DOI's redirect chain to the latest record ID."""
    resp = requests.get(
        f"https://doi.org/{concept_doi}", allow_redirects=True, timeout=30
    )
    resp.raise_for_status()
    record_id = resp.url.rstrip("/").rsplit("/", 1)[-1]
    logger.info("Resolved concept DOI %s -> latest record %s", concept_doi, record_id)
    return record_id


def _missing_files(data_dir: Path) -> list[str]:
    return [f for f in REQUIRED_FILES if not (data_dir / f).exists()]


def ensure_data_available(
    data_dir: Path,
    concept_doi: str = ZENODO_CONCEPT_DOI,
    archive_name: str = "data.zip",
) -> None:
    """Download and extract the Zenodo archive if any required file is missing."""
    missing = _missing_files(data_dir)
    if not missing:
        logger.info("All required data files present in %s.", data_dir)
        return

    logger.warning(
        "Missing %d required file(s) in %s: %s. Attempting download from Zenodo...",
        len(missing),
        data_dir,
        missing,
    )

    record_id = _resolve_latest_record_id(concept_doi)
    download_url = (
        f"https://zenodo.org/api/records/{record_id}/files/{archive_name}/content"
    )

    extract_to = data_dir.parent
    extract_to.mkdir(parents=True, exist_ok=True)
    zip_path = extract_to / archive_name

    try:
        response = requests.get(download_url, stream=True, timeout=120)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded %s (%d bytes).", download_url, zip_path.stat().st_size)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info("Extracted data archive to %s.", extract_to)

    except requests.exceptions.RequestException as e:
        logger.error("Failed to download data from %s: %s", download_url, e)
        raise
    except zipfile.BadZipFile as e:
        logger.error("Failed to unzip downloaded archive: %s", e)
        raise
    finally:
        zip_path.unlink(missing_ok=True)

    still_missing = _missing_files(data_dir)
    if still_missing:
        raise FileNotFoundError(
            f"Data still incomplete after download: missing {still_missing}. "
            f"Check the Zenodo record at https://doi.org/{concept_doi}."
        )
