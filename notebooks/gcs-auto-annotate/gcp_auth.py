"""Google Cloud credentials for the scripts.

Uses Application Default Credentials when available (``gcloud auth application-default login``
or a service account via GOOGLE_APPLICATION_CREDENTIALS).  If none are configured it falls back
to the token of your normal ``gcloud auth login`` session, so a plain ``gcloud auth login`` is
enough to try the demo.
"""
import subprocess
import threading
import time

import google.auth
import google.auth.transport.requests
from google.oauth2.credentials import Credentials

_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
_REFRESH_AFTER = 30 * 60  # seconds; gcloud tokens last about an hour


class _GcloudToken:
    """Credentials backed by `gcloud auth print-access-token`, refreshed on demand."""

    def __init__(self):
        self._lock = threading.Lock()
        self._token = None
        self._fetched = 0.0

    def token(self):
        with self._lock:
            if self._token is None or time.time() - self._fetched > _REFRESH_AFTER:
                try:
                    out = subprocess.run(
                        ["gcloud", "auth", "print-access-token"],
                        check=True, capture_output=True, text=True).stdout.strip()
                except (OSError, subprocess.CalledProcessError) as exc:
                    raise SystemExit(
                        "No Google credentials found. Run `gcloud auth login` "
                        f"(or `gcloud auth application-default login`) and retry. ({exc})")
                self._token, self._fetched = out, time.time()
            return self._token


_gcloud = _GcloudToken()
_adc = None
_adc_checked = False
_adc_lock = threading.Lock()


def _application_default():
    global _adc, _adc_checked
    with _adc_lock:
        if not _adc_checked:
            _adc_checked = True
            try:
                _adc, _ = google.auth.default(scopes=_SCOPES)
            except google.auth.exceptions.DefaultCredentialsError:
                _adc = None
        return _adc


def access_token():
    """A currently valid OAuth access token."""
    adc = _application_default()
    if adc is not None:
        with _adc_lock:
            if not adc.valid:
                adc.refresh(google.auth.transport.requests.Request())
            return adc.token
    return _gcloud.token()


def auth_headers():
    return {"Authorization": f"Bearer {access_token()}"}


def storage_client():
    """A google-cloud-storage client using the credentials above."""
    from google.cloud import storage
    adc = _application_default()
    if adc is not None:
        return storage.Client(credentials=adc)
    return storage.Client(credentials=Credentials(_gcloud.token()), project="unused")
