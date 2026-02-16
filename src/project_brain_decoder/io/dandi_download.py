import requests
from nwb_loader import get_project_root


# Example data sessions from the DANDI archive:
SESSION_URLS = {
    "session1": ("https://api.dandiarchive.org/api/assets/c002a9a1-664d-4a69-af02-ba810046c4fb/download/",
                 "sub-Monkey-N_ses-20200127_ecephys.nwb"),
    "session2": ("https://api.dandiarchive.org/api/assets/9d1820f1-7583-4faf-bbd0-7e9fb7001ca4/download/",
                 "sub-Monkey-N_ses-20200130_ecephys.nwb"),
    "session3": ("https://api.dandiarchive.org/api/assets/ea07a2e3-d5f4-4036-9b62-93d1f89cba64/download/",
                 "sub-Monkey-N_ses-20200204_ecephys.nwb"),
    "session4": ("https://api.dandiarchive.org/api/assets/24b91bc3-92ba-457c-8718-490b508f6b8f/download/",
                 "sub-Monkey-N_ses-20200205_ecephys.nwb"),
    "session5": ("https://api.dandiarchive.org/api/assets/008fc384-2a95-47ec-8dbe-7acb4a18bcc7/download/",
                 "sub-Monkey-N_ses-20200206_ecephys.nwb"),
    "session6": ("https://api.dandiarchive.org/api/assets/9c0ac931-d97e-464b-a134-c366b7c84727/download/",
                 "sub-Monkey-N_ses-20200211_ecephys.nwb"),
    "session7": ("https://api.dandiarchive.org/api/assets/19a1f617-abb2-45ed-81d5-7f09dc60890d/download/",
                 "sub-Monkey-N_ses-20200222_ecephys.nwb"),
    "session8": ("https://api.dandiarchive.org/api/assets/ed7d5522-a1f8-4725-948d-8f05c884a22f/download/",
                 "sub-Monkey-N_ses-20200224_ecephys.nwb"),
    "session9": ("https://api.dandiarchive.org/api/assets/15299ea6-cffb-46e8-9175-a822e4f52f1d/download/",
                 "sub-Monkey-N_ses-20200225_ecephys.nwb"),
    "session10": ("https://api.dandiarchive.org/api/assets/b424f116-7827-4ab0-80ed-1e8951eea67a/download/",
                  "sub-Monkey-N_ses-20200228_ecephys.nwb"),
    "session11": ("https://api.dandiarchive.org/api/assets/d6d75bfa-fe24-4d2b-a3fc-13e11fe00424/download/",
                  "sub-Monkey-N_ses-20200302_ecephys.nwb"),
    "session12": ("https://api.dandiarchive.org/api/assets/97ca90b1-12e8-46f8-8f5a-11a42b0cdda8/download/",
                  "sub-Monkey-N_ses-20200310_ecephys.nwb")
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
        path = root / "data" / "preprocessed" / url[1]
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.as_posix(), "wb") as f:
            chunk_size = 4*1024*1024
            for chunk in r.iter_content(chunk_size):
                if chunk:
                    f.write(chunk)
        return


download_session(url=SESSION_URLS["session1"])