from typing import Any, Dict, List, Optional, Tuple
from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.video_utils import read_video_timestamps
from torchvision.io.video import read_video
import tempfile
import os
from torch.utils.data.dataloader import DataLoader
import torch

from aperturedb.Videos import Videos
from tqdm import tqdm

class _VideoTimestampsDataset:
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.

    Used in VideoClips and defined at top level so it can be
    pickled when forking.
    """
    def __init__(self, videos: Videos) -> None:
        self._videos = videos
        self._tmp_path = "scratch"
        if os.path.exists(self._tmp_path) and os.path.isdir(self._tmp_path):
            pass


    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[float]]:
        video = self._videos[idx]

        with tempfile.NamedTemporaryFile(dir=self._tmp_path, suffix=".mp4") as ostream:
            ostream.write(video["preview"])
            x = read_video_timestamps(ostream.name)
            return x
        raise Exception("Should not be here")

class AVideoClips(VideoClips):
    """
    Pytorch VideoClips with aperturedb.
    """
    def __init__(self, videos: Videos, clip_length_in_frames: int = 16, frames_between_clips: int = 1,
        frame_rate: Optional[int] = None, _precomputed_metadata: Optional[Dict[str, Any]] = None, num_workers: int = 0,
        _video_width: int = 0, _video_height: int = 0, _video_min_dimension: int = 0, _video_max_dimension: int = 0,
        _audio_samples: int = 0, _audio_channels: int = 0, output_format: str = "THWC") -> None:
        self._videos = videos
        self._num_workers = num_workers

        # these options are not valid for pyav backend
        self._video_width = _video_width
        self._video_height = _video_height
        self._video_min_dimension = _video_min_dimension
        self._video_max_dimension = _video_max_dimension
        self._audio_samples = _audio_samples
        self._audio_channels = _audio_channels
        self.output_format = output_format.upper()

        self._compute_frame_pts()
        self.compute_clips(clip_length_in_frames, frames_between_clips, frame_rate)
        assert len(self._videos) == len(list(filter(lambda e: 'preview' in e, self._videos)))




    def _compute_frame_pts(self) -> None:
        dl: DataLoader = DataLoader(
            _VideoTimestampsDataset(self._videos),
            batch_size=16,
            num_workers=self._num_workers,
            collate_fn=lambda x: x
            )

        self.video_fps = []
        self.video_pts = []

        with tqdm(total=len(dl)) as pbar:
            for batch in dl:
                pbar.update(1)
                clips, fps = list(zip(*batch))
                clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
                self.video_pts.extend(clips)
                self.video_fps.extend(fps)

    def __len__(self) -> int:
        return len(self._videos)

    def get_clip(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any], int]:
        """
        Gets a subclip from a list of videos.

        Args:
            idx (int): index of the subclip. Must be between 0 and num_clips().

        Returns:
            video (Tensor)
            audio (Tensor)
            info (Dict)
            video_idx (int): index of the video in `video_paths`
        """
        if idx >= self.num_clips():
            raise IndexError(f"Index {idx} out of range ({self.num_clips()} number of clips)")
        video_idx, clip_idx = self.get_clip_location(idx)
        clip_pts = self.clips[video_idx][clip_idx]

        from torchvision import get_video_backend

        backend = get_video_backend()

        if backend == "pyav":
            # check for invalid options
            if self._video_width != 0:
                raise ValueError("pyav backend doesn't support _video_width != 0")
            if self._video_height != 0:
                raise ValueError("pyav backend doesn't support _video_height != 0")
            if self._video_min_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_min_dimension != 0")
            if self._video_max_dimension != 0:
                raise ValueError("pyav backend doesn't support _video_max_dimension != 0")
            if self._audio_samples != 0:
                raise ValueError("pyav backend doesn't support _audio_samples != 0")

        if backend == "pyav":
            start_pts = clip_pts[0].item()
            end_pts = clip_pts[-1].item()
            with tempfile.NamedTemporaryFile(dir="scratch", suffix=".mp4") as ostream:
                ostream.write(self._videos[video_idx]["preview"])
                video, audio, info = read_video(ostream.name, start_pts, end_pts)

        if self.frame_rate is not None:
            resampling_idx = self.resampling_idxs[video_idx][clip_idx]
            if isinstance(resampling_idx, torch.Tensor):
                resampling_idx = resampling_idx - resampling_idx[0]
            video = video[resampling_idx]
            info["video_fps"] = self.frame_rate
        assert len(video) == self.num_frames, f"{video.shape} x {self.num_frames}"

        if self.output_format == "TCHW":
            # [T,H,W,C] --> [T,C,H,W]
            video = video.permute(0, 3, 1, 2)

        return video, audio, info, video_idx


