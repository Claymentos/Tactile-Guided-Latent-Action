import torch
from lam.datasets.egodex.utils.skeleton_tfs import WRISTS, TWO_HANDS_TFS
from lam.datasets.egodex.utils.data_utils import *
from torchcodec.decoders import VideoDecoder
import torchvision.transforms.functional as F

class EgoDexDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_dir,
                 len_seq,
                 img_size,
                 query_tfs=TWO_HANDS_TFS):
        super().__init__()
        self.img_size = img_size
        self.len_seq = len_seq
        self.data_dir = data_dir
        self.query_tfs = query_tfs
        self.dataset_path_list, self.episode_len = index_episodes(self.data_dir)
        self.cumulative_len = np.cumsum(self.episode_len)

    def __len__(self):
        return sum([length - self.len_seq for length in self.episode_len if length > self.len_seq])    

    def locate_transition(self, index):
            adjusted_episode_lens = [length - self.len_seq for length in self.episode_len]
            adj_cumulative_len = np.cumsum(adjusted_episode_lens)
            
            episode_index = np.argmax(adj_cumulative_len > index)
            if episode_index == 0:
                start_frame = index
            else:
                start_frame = index - adj_cumulative_len[episode_index - 1]
                
            return episode_index, start_frame

    def __getitem__(self, idx):
            episode_id, frame_id = self.locate_transition(idx)
            hdf5_file = self.dataset_path_list[episode_id]
            mp4_file = hdf5_file[:-5] + '.mp4'

            with h5py.File(hdf5_file, 'r') as root:
                tfs_seq = []
                for t in range(frame_id, frame_id + self.len_seq):
                    tfs = np.zeros([len(self.query_tfs), 4, 4])
                    for i, tf_name in enumerate(self.query_tfs):
                        tfs[i] = root['/transforms/' + tf_name][t]
                    tfs_seq.append(tfs)
                
                tfs_seq = np.array(tfs_seq) 
                cam_int = root['/camera/intrinsic'][:]
                lang_instruct = root.attrs['llm_description'] 

            decoder = VideoDecoder(mp4_file, device='cpu')
            img_seq = []
            for t in range(frame_id, frame_id + self.len_seq):
                img = decoder[t] 
                img = F.resize(img, self.img_size)
                img = img.float() / 255.0
                img_seq.append(img)
            
            video_tensor = torch.stack(img_seq) 

            return {
                "images": video_tensor,     
                "actions": tfs_seq,        
                "instr": lang_instruct
            }