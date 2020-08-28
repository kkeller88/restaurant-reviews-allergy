import os
import shutil
import tempfile

import mlflow

class MlflowArtifactLogger(object):
    def __init__(self, base=None, cleanup_after=True):
        self.cleanup_after = cleanup_after
        if base is None:
            self.base = tempfile.mkdtemp()
            #os.makedirs(self.base)
        else:
            self.base = base

    def add_artifact(self, artifact, name, format='pkl'):
        path = os.path.join(self.base, name)
        if format=='pkl':
            artifact.to_pickle(path)
            print('Artifact written to: ', path)
        else:
            message = 'format {f} was not recognized'.format(f=format)
            raise ValueError(message)

    def log_artifacts(self, path):
        mlflow.log_artifacts(local_dir=self.base, artifact_path=path)
        if self.cleanup_after:
            shutil.rmtree(self.base)
        message = 'Artifacts from {b} logged to mlflow!'.format(b=self.base)
        print(message)
