"""S3-compatible object storage — upload files and return public URLs."""

from __future__ import annotations

import logging
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from daily_research.config import Settings

log = logging.getLogger(__name__)


def _get_s3_client(settings: Settings):
    """Create a boto3 S3 client from settings."""
    return boto3.client(
        "s3",
        endpoint_url=settings.s3_api_url or None,
        aws_access_key_id=settings.s3_access_key_id,
        aws_secret_access_key=settings.s3_secret_access_key,
        config=BotoConfig(signature_version="s3v4"),
    )


def upload_to_s3(
    file_path: Path,
    *,
    settings: Settings,
    key: str | None = None,
    content_type: str = "application/pdf",
) -> str | None:
    """Upload a file to the configured S3 bucket and return its public URL.

    Parameters
    ----------
    file_path:
        Local path to the file to upload.
    settings:
        Application settings containing S3 credentials.
    key:
        Object key (path within the bucket).  Defaults to
        ``reports/<filename>``.
    content_type:
        MIME type for the uploaded object.

    Returns
    -------
    str | None
        The public URL of the uploaded object, or *None* if S3 is not
        configured.
    """
    if not settings.s3_api_url or not settings.s3_bucket_name:
        log.warning("S3 not configured — skipping PDF upload.")
        return None

    if key is None:
        key = f"reports/{file_path.name}"

    client = _get_s3_client(settings)

    try:
        client.upload_file(
            str(file_path),
            settings.s3_bucket_name,
            key,
            ExtraArgs={
                "ContentType": content_type,
                "ACL": "public-read",
            },
        )
    except Exception as exc:
        log.error("S3 upload failed: %s", exc)
        return None

    # Build public URL
    if settings.s3_public_url:
        base = settings.s3_public_url.rstrip("/")
        if not base.startswith("http"):
            base = f"https://{base}"
        url = f"{base}/{key}"
    else:
        api_url = settings.s3_api_url.rstrip("/")
        url = f"{api_url}/{settings.s3_bucket_name}/{key}"

    log.info("Uploaded to S3 → %s", url)
    return url
