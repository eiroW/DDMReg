# %%
import torch
import numpy as np
from matplotlib import pyplot as plt
import os
os.environ['VXM_BACKEND'] = 'pytorch'
import ddmreg
import glob

# %%

model = ddmreg.networks.DDMReg_dMRI_TOMs_reorient_mem.load(os.path.join('ddmreg_models', 'ddmreg_model_fa_ep0750.pt'), 'cpu')

# %%
targetDir = 'test/sub_1/'
movingDir = 'test/sub_2/'

# %%
target_fa   = sorted(glob.glob(os.path.join(targetDir, '*_fa.nii.gz')))
moving_fa   = sorted(glob.glob(os.path.join(movingDir, '*_fa.nii.gz')))

# %%
target_vol = torch.from_numpy(ddmreg.utils.load_volfile(target_fa[0]))
moving_vol = torch.from_numpy(ddmreg.utils.load_volfile(moving_fa[0]))


# %%
flow  = torch.load('test/sub_1-TO-sub_2/flow.pt',map_location=torch.device('cpu'))

# %%
flow.shape

# %%
target_vol.shape

# %%
fa_wraped = ddmreg.utils.load_volfile('test/sub_1-TO-sub_2/fa_warped.nii.gz')


model.tract_tom_model_names = None

model.eval()
y_src,flow,_,_ = model(moving_vol.unsqueeze(0).unsqueeze(0),target_vol.unsqueeze(0).unsqueeze(0))

with torch.no_grad():
    flow1 = model.fullsize(flow)
    y_src1=torch.cat([y_src for _ in range(3)],dim=1)
    a = model.reorienter(y_src1,flow1)

# %%
plt.imshow(np.flip(a.squeeze().detach().numpy()[1,...,50].T), cmap='gray')

