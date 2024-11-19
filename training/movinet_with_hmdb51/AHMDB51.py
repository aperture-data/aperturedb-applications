from typing import Any, Callable, Dict, Optional
from aperturedb.Videos import Videos
from aperturedb.CommonLibrary import create_connector, execute_query
import torchvision
import AVideoClips


def get_videos(train:bool, split:int) -> Videos:
    """
    HMDB51 stores videos in clips corresponding to 51 categories.
    They videos are classified as a test and train set (70% : 30%)

    The data set is further stored in 3 ways,
    Get videos from aperturedb based on type (Train/Test)
    and split.

    Fetch the appropriate set.
    """

    client = create_connector()

    query = [{
        "FindEntity": {
            "_ref": 1,
            "with_class": "Split",
            "constraints": {
                "id": ["==", split]
            },
            "results": {
                "all_properties": True
            }
        }
    }, {
        "FindVideo":{
            "is_connected_to": {
                "ref": 1,
                "constraints": {
                    "type": ["==", 1 if train else 2]
                }
            },
            "results":{
                "all_properties": True,
                "count": True
            }
        }
    }]
    _, r, b = execute_query(client, query, [])


    videos = Videos(client=client, response=r[1]["FindVideo"]["entities"])
    videos.blobs = True
    print(f"Retrieved {len(videos)} videos")
    return videos


class AHMDB51(torchvision.datasets.HMDB51):
    """
    Implementation of HMDB51 aware of aperturedb.
    Notice how pytorch's implementation has so much code for local file processing.
    """
    def __init__(self,
        frames_per_clip: int = 5,
        step_between_clips: int = 1,
        frame_rate: Optional[int] = None,
        fold: int = 1, train: bool = True,
        transform: Optional[Callable] = None,
        _precomputed_metadata: Optional[Dict[str, Any]] = None,
        num_workers: int = 1,
        _video_width: int = 0,
        _video_height: int = 0,
        _video_min_dimension: int = 0,
        _audio_samples: int = 0,
        output_format: str = "THWC") -> None:
        self.video_pts = []
        self.video_fps = []
        self.transform = transform

        videos = get_videos(train=train, split=fold)
        self.ci = {}
        videos.blobs = False
        for v in videos:
            if v["category"] not in self.ci:
                self.ci[v["category"]] = len(self.ci)
        self.samples = [(i, self.ci[v["category"]]) for i, v in enumerate(videos)]
        videos.blobs = True


        video_clips = AVideoClips(
            videos,
            frames_per_clip,
            step_between_clips,
            frame_rate,
            _precomputed_metadata,
            num_workers=num_workers,
            _video_width=_video_width,
            _video_height=_video_height,
            _video_min_dimension=_video_min_dimension,
            _audio_samples=_audio_samples,
            output_format=output_format,
        )

        self.video_clips = video_clips
        self.indices = [i for i in range(len(videos))]
        assert len(videos) == len(list(filter(lambda e: 'preview' in e, videos)))
        videos.loaded = True