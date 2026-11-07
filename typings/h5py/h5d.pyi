"""The datasets of h5py's low-level API that mosaic uses."""

from .h5g import GroupID
from .h5p import PropDCID
from .h5s import SpaceID
from .h5t import TypeID

class DatasetID:
    def get_create_plist(self) -> PropDCID: ...

def create(loc: GroupID, name: bytes, tid: TypeID, space: SpaceID) -> DatasetID: ...
