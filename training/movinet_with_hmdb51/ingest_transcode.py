import os
import subprocess

from aperturedb.ParallelLoader import ParallelLoader
from aperturedb.QueryGenerator import QueryGenerator
from aperturedb.CommonLibrary import create_connector
from aperturedb.Utils import Utils


class TreeIngest(QueryGenerator):
    def __init__(self, root_path: str, annotation_path: str) -> None:
        self._files = [os.path.join(dirpath, f) for dirpath, dirs, filenames in os.walk(root_path) for f in filenames if f.endswith("avi")]
        self._fc = {os.path.basename(file_path): file_path for file_path in self._files}
        print(f"{len(self._files)=}, {len(self._fc)=}")

        self._items = []
        for _,_, filenames in os.walk(annotation_path):
            for filename in filenames:
                with open(os.path.join(annotation_path, filename), "r") as ins:
                    split = int(filename.split(".")[0][-1])
                    lines = ins.readlines()
                    for line in lines:
                        path, code = line.split()
                        if path in self._fc:
                            if os.path.exists(self._fc[path]):
                                self._items.append((split, int(code), self._fc[path]))
                        else:
                            print(f"{path} from splits not in files")
        print("processed")

    def __repr__(self) -> str:
        return f"A collection of {len(self)} smaples"

    def __len__(self):
        return len(self._items)

    def getitem(self, subscript):
        split, code, file_path = self._items[subscript]
        dest_path = file_path.replace(".avi", ".mp4")
        if not os.path.exists(dest_path):
            p = subprocess.Popen(
                f"ffmpeg -i '{file_path}' -vcodec libx264 -acodec aac '{dest_path}' 1> /dev/null 2>/dev/null",
                shell=True,
            )
            out, err = p.communicate()
            if out or err:
                print(f"res: {out, err}")
                if "error" in err:
                    print(f"Error transcoding {file_path}")
                    return None
        category = file_path.split("/")[-2]
        video_uid = os.path.basename(dest_path)
        connection_uid  = f"{os.path.basename(dest_path)}_{split}_{code}"
        query = [
            {
                "AddEntity": {
                    "_ref": 1,
                    "class": "Split",
                    "properties": {
                        "id": split
                    },
                    "if_not_found": {
                        "id": ["==", split]
                    }
                }
            },
            {
                "AddVideo": {
                    "_ref": 2,
                    "properties": {
                        "name": video_uid,
                        "category": category
                    },
                    "if_not_found": {
                        "name": ["==", video_uid]
                    }
                }
            },
            {
                "AddConnection":{
                    "class": "IsInSplit",
                    "src": 2,
                    "dst": 1,
                    "properties": {
                        "type": code,
                        "id": connection_uid
                    },
                    "if_not_found": {
                        "id": ["==", connection_uid]
                    }
                }
            }
        ]
        buffer = None
        with open(dest_path, "rb") as instream:
            buffer = instream.read()
        return query, [buffer]

if __name__ == "__main__":



    generator = TreeIngest("input/categories", "splits/testTrainMulti_7030_splits")
    print(generator)

    # Create a client.
    client = create_connector()

    utils = Utils(client)
    assert utils.create_entity_index("Split", "id"), "Failed to create index for Split"
    assert utils.create_entity_index("_Video", "name"), "Failed to create index for _Video"
    assert utils.create_connection_index("IsInSplit", "id"), "Failed to create index for IsInSplit"

    # Create a loader
    loader = ParallelLoader(client=client, dry_run=False)

    # Ingest the data
    loader.ingest(generator=generator, batchsize=1, stats=True)
