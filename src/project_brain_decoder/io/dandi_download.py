import requests
from pathlib import Path
from nwb_loader import get_project_root


# Example data sessions from the DANDI archive:
SESSION_URLS = {
    "session1": ("https://api.dandiarchive.org/api/assets/c002a9a1-664d-4a69-af02-ba810046c4fb/download/",
                 "sub-Monkey-N_ses-20200127_ecephys.nwb"),
    "session2": ("https://api.dandiarchive.org/api/assets/9d1820f1-7583-4faf-bbd0-7e9fb7001ca4/download/",
                 "sub-Monkey-N_ses-20200130_ecephys.nwb"),
    "session3": ("https://api.dandiarchive.org/api/assets/ea07a2e3-d5f4-4036-9b62-93d1f89cba64/download/",
                 "sub-Monkey-N_ses-20200204_ecephys.nwb")
}


def download_session(url: tuple[str, str]):
    """
    Downloading the session dataset in chunks and saving it as an NWB file
    :param url: tuple[str, str]
    :return:
    """
    with requests.get(url[0], stream=True) as r:
        r.raise_for_status()
        root = get_project_root()
        path = root / "data" / "raw" / url[1]
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.as_posix(), "wb") as f:
            chunk_size = 4*1024*1024
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        return


download_session(url=SESSION_URLS["session3"])