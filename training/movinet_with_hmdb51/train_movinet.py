import time
import torchvision
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import v2 as T
from movinets import MoViNet
from movinets.config import _C



from torch.utils.data.dataloader import DataLoader
from typer import Typer
import AHMDB51


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
def inference(use_aperturedb:bool):
    train_loader, test_loader = get_data_loaders(use_aperturedb=use_aperturedb)
    print(test_loader.dataset.ci)

@app.command()
def training(use_aperturedb:bool):
    train_loader, test_loader = get_data_loaders(use_aperturedb=use_aperturedb)
    train(train_loader=train_loader, test_loader=test_loader)
    classes = test_loader.dataset.ci
    rc = {v:k for k,v in classes.items()}

    # Preserve the classes to index mapping for this model.
    import json
    with open("classes.json", "w") as out:
        json.dump(rc, out, indent=2)

if __name__ == "__main__":
   app()