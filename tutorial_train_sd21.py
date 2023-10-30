from share import *
import sys
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict


# Configs
resume_path = './models/control_sd21_ini.ckpt'
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False
if len(sys.argv) > 1:
    batch_size = sys.argv[1]
    config_path = './models/cldm_v21.yaml'
else:
    batch_size = 2
    config_path = './models/cldm_v21_reduced.yaml'


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model(config_path).cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
