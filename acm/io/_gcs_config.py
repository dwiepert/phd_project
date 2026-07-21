"""
Google Cloud Storage object to handle upload/download

Author(s): Daniela Wiepert
Last modified: 06/2026
"""
#TODO: Test this
#IMPORTS
##built-in
from typing import List, Union 
from pathlib import Path

##third-party
from google.cloud import storage

class GCSConfig():
    def __init__(self, bucket_name:str, project_name:str):
        """
        Docstring for __init__
        
        :param self: Description
        :param bucket_name: Description
        :type bucket_name: str
        :param project_name: Description
        :type project_name: str
        """
        super(GCSConfig, self).__init__()
        self._bucket_name = bucket_name
        self._project_name = project_name 

        storage_client = storage.Client(project=self._project_name)
        self.bucket = storage_client.bucket(self._bucket_name)

    def get_blob(self, prefix:Union[str, Path]):
        """
        Get GCS blob
        
        :param prefix: str, GCS prefix
        :return: blob for prefix
        """
        assert (isinstance(prefix, str) or isinstance(prefix, Path)), 'Must give path-like object for GCS prefix.'
        return self.bucket.blob(str(prefix))

    def download(self, prefix:Union[str, Path], local_path: Union[str, Path]) -> List[Path]:
        """
        Download file(s) to local path

        :param prefix: path-like, GCS prefix
        :param local_path: path-like, desired local save path
        :return: List[Path], absolute savepaths for all files downloaded to local directory
        """
        local_path = Path(local_path).absolute()
        # if is_directory:
        #:param is_directory: bool, true if prefix is a directory and not absolute path
        #     files = self.search("*", prefix)
        # else:
        #TODO: check this works for directories too
        pattern = str(Path(prefix).name)
        parent_prefix = Path(prefix).parents[0]
        files = self.search(pattern, parent_prefix, exact_match=True)

        #prune files of directories
        files = [f for f in files if Path(f).suffix != '']

        #handling weirdness if "/" was given at the end of the prefix
        str_prefix = str(prefix)
        if str_prefix[-1] != "/": str_prefix += "/"
        paths = []

        for f in files: 
            sub_path = local_path / f.replace(str_prefix, "")
            sub_path.parents[0].mkdir(parents=True, exist_ok=True)

            blob = self.bucket.blob(f)
            blob.download_to_filename(str(sub_path))
            paths.append(sub_path)
        
        return paths
    
    def upload(self, local_path:Union[str, Path], prefix:Union[str,Path], overwrite:bool=False) -> List[Path]:
        """
        Upload file(s) to GCS 
        
        :param local_path: path-like, desired local save path
        :param prefix: path-like, GCS prefix
        :param overwrite: bool, True if overwriting existing file
        :return to_upload: List[Path] list of uploaded files with their new GCS paths
        """
        if not isinstance(local_path, Path): local_path = Path(local_path)
        if not isinstance(prefix, Path): prefix = Path(prefix)
        if local_path.is_dir():
            pattern = "*"
            to_upload = [r for r in local_path.rglob(pattern) if not r.is_dir()]
        else:
            pattern = str(local_path.name)
            to_upload = [local_path]

        existing = self.search(pattern, prefix)

        if not overwrite:
            keep = []
            for r in to_upload:
                if not any([str(r.name) in e for e in existing ]):
                    keep.append(r)
            to_upload = keep

        for u in to_upload:
            if len(to_upload) > 100:
                ind = to_upload.index(u)
                if ind % 100 == 0: print(f'{ind}/{len(to_upload)}')
            name = str(u).split("/")
            i = len(name) - 1
            to_add = []
            if any([n == prefix.name for n in name]):
                while (name[i] != prefix.name and i >= 0):
                    to_add.append(name[i])
                    i -= 1
                to_add.reverse()
                new_name = "/".join(to_add)
            else:
                new_name = u.name

            blob = self.bucket.blob(str(prefix / new_name))
            blob.upload_from_filename(str(u))

        return to_upload 
    
    def exists(self, prefix):
        """
        Check if a path exists in GCS bucket 

        :param prefix: path-like, GCS prefix
        """
        #TODO: check functionality
        existing = self.search(prefix, prefix, True)
        if existing:
            return True
        else:
            return False 
        
    def search(self, pattern:str, prefix:Union[str, Path], exact_match:bool=False) -> List[str]:
        """
        Search GCS bucket for specific file(s)

        :param pattern: str, pattern to search for
        :param prefix: path-like, GCS prefix
        :param exact_match: bool, True if looking for exact match (default = False)
        :return files: list of files
        """
        assert isinstance(pattern, str), 'Must give string pattern for searching GCS bucket.'
        assert (isinstance(prefix, str) or isinstance(prefix, Path)), 'Must give path-like object for GCS prefix.'

        files = []
        blobs = self.bucket.list_blobs(prefix=prefix)
        for blob in blobs:
            if exact_match:
                s_name = Path(blob.name).parts
                if any([s == pattern for s in s_name]) or pattern == blob.name: #either a subpart matches or it's a full path match
                    files.append(blob.name)
            else:
                if (pattern in blob.name) or (pattern == '*' and blob.name != prefix):
                    files.append(blob.name)

        return files