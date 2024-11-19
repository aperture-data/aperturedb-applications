import time
from typing import Any, Callable, Dict, List, Optional, Tuple
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2 as T
from movinets import MoViNet
from movinets.config import _C

from aperturedb.Videos import Videos
from aperturedb.CommonLibrary import create_connector, execute_query
import tempfile

from torchvision.datasets.video_utils import VideoClips
from torchvision.datasets.video_utils import read_video_timestamps
from torchvision.io.video import read_video

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typer import Typer


class _VideoTimestampsDataset:
    """
    Dataset used to parallelize the reading of the timestamps
    of a list of videos, given their paths in the filesystem.

    Used in VideoClips and defined at top level so it can be
    pickled when forking.
    """

    def __init__(self, videos: Videos) -> None:
        self._videos = videos

    def __len__(self) -> int:
        return len(self._videos)

    def __getitem__(self, idx: int) -> Tuple[List[int], Optional[float]]:
        video = self._videos[idx]
        with tempfile.NamedTemporaryFile(dir="scratch", suffix=".mp4") as ostream:
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

def get_common():
    """
    Just common parameters.
    Applies to the training and data loading sections.
    """
    torch.manual_seed(97)
    num_frames = 16 # 16
    clip_steps = 2
    Bs_Train = 16
    Bs_Test = 16

    transform = T.Compose([
                                    T.Lambda(lambda x: x.permute(3, 0, 1, 2) / 255.),
                                    T.Resize((200, 200)),
                                    T.RandomHorizontalFlip(),
                                    # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.RandomCrop((172, 172))])
    transform_test = T.Compose([
                                    T.Lambda(lambda x: x.permute(3, 0, 1, 2) / 255.),
                                    # T.ToTensor()/255.0,
                                    # T.ToTensor(),
                                    T.Resize((200, 200)),
                                    # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
                                    T.CenterCrop((172, 172))])
    return num_frames, clip_steps, Bs_Train, Bs_Test, transform, transform_test

def get_local_data_sets():
    """
    Build datasets from local files.
    This is the original code.
    """
    num_frames, clip_steps, Bs_Train, Bs_Test, transform, transform_test = get_common()
    hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,frame_rate=5,
                                                    step_between_clips = clip_steps, fold=1, train=True,
                                                    transform=transform, num_workers=1)

    hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames,frame_rate=5,
                                                    step_between_clips = clip_steps, fold=1, train=False,
                                                    transform=transform_test, num_workers=1)
    return hmdb51_train, hmdb51_test

def get_data_sets():
    """
    Get the datasets from aperturedb.
    The data has been ingested previously.
    """
    num_frames, clip_steps, Bs_Train, Bs_Test, transform, transform_test = get_common()

    hmdb51_train = AHMDB51(
        num_workers=1,
        frame_rate=5,
        frames_per_clip=num_frames,
        step_between_clips=clip_steps,
        train=True,
        transform=transform
        )
    hmdb51_test = AHMDB51(
        num_workers=1,
        frame_rate=5,
        frames_per_clip=num_frames,
        step_between_clips=clip_steps,
        train=False,
        transform=transform_test
        )


    return hmdb51_train, hmdb51_test

def get_data_loaders(use_aperturedb: bool=False):
    """
    Build Data loaders using the datasets
    arg use_aperturedb defines how to get datasets
    """
    num_frames, clip_steps, Bs_Train, Bs_Test, transform, transform_test = get_common()
    if not use_aperturedb:
        hmdb51_train, hmdb51_test = get_local_data_sets()
    else:
        hmdb51_train, hmdb51_test = get_data_sets()



    train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
    test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)
    return train_loader, test_loader

def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    # model.cuda()
    model.cpu()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data,_ , target) in enumerate(data_load):
        # out = F.log_softmax(model(data.cuda()), dim=1)
        out = F.log_softmax(model(data.cpu()), dim=1)
        # loss = F.nll_loss(out, target.cuda())
        loss = F.nll_loss(out, target.cpu())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())

def evaluate(model, data_load, loss_val):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            # output = F.log_softmax(model(data.cuda()), dim=1)
            output = F.log_softmax(model(data.cpu()), dim=1)
            # loss = F.nll_loss(output, target.cuda(), reduction='sum')
            loss = F.nll_loss(output, target.cpu(), reduction='sum')
            _, pred = torch.max(output, dim=1)

            tloss += loss.item()
            # csamp += pred.eq(target.cuda()).sum()
            csamp += pred.eq(target.cpu()).sum()

            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

def train_iter_stream(model, optimz, data_load, loss_val, n_clips = 2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    #clean the buffer of activations
    samples = len(data_load.dataset)
    # model.cuda()
    model.cpu()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    for i, (data,_, target) in enumerate(data_load):
        # data = data.cuda()
        # target = target.cuda()
        data = data.cpu()
        target = target.cpu()
        l_batch = 0
        #backward pass for each clip
        for j in range(n_clips):
          output = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
          loss = F.nll_loss(output, target)
          _, pred = torch.max(output, dim=1)
          loss = F.nll_loss(output, target)/n_clips
          loss.backward()
        l_batch += loss.item()*n_clips
        optimz.step()
        optimz.zero_grad()

        #clean the buffer of activations
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)

def evaluate_stream(model, data_load, loss_val, n_clips = 2, n_clip_frames=8):
    model.eval()
    # model.cuda()
    model.cpu()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in data_load:
            # data = data.cuda()
            # target = target.cuda()
            data = data.cpu()
            target = target.cpu()
            model.clean_activation_buffers()
            for j in range(n_clips):
              output = F.log_softmax(model(data[:,:,(n_clip_frames)*(j):(n_clip_frames)*(j+1)]), dim=1)
              loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss /  len(data_load)
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')

def train(train_loader, test_loader):
    N_EPOCHS = 1

    # Use the original movinet based on Kinetics400 dataset when we get pretrained.
    model = MoViNet(_C.MODEL.MoViNetA0, causal = False, pretrained = True )
    start_time = time.time()

    trloss_val, tsloss_val = [], []
    model.classifier[3] = torch.nn.Conv3d(2048, 51, (1,1,1))
    optimz = optim.Adam(model.parameters(), lr=0.00005)
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_iter(model, optimz, train_loader, trloss_val)
        evaluate(model, test_loader, tsloss_val)
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimz.state_dict(),
            'epoch': epoch,
            'train_loss': trloss_val,
            'test_loss': tsloss_val
        }, f'movinet_{epoch}.pth')

        # Save every epoch and can compare across epochs too. Right now we stop at 1
        # HMDB51 has 6000 clips with 51 classes.
        # This trains on a split based on fold value selected when we load dataset
        torch.save(model, f'movinet_hmdb51_{epoch}.pth')

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')



app = Typer()

@app.command()
def inference():
    pass

@app.command()
def training(use_aperturedb:bool):
    train_loader, test_loader = get_data_loaders(use_aperturedb=use_aperturedb)
    train(train_loader=train_loader, test_loader=test_loader)

if __name__ == "__main__":
   app()