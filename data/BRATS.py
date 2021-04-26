from os import listdir
from os.path import join, isfile
import random
from scipy import io as sio
import numpy as np
import copy
from ipdb import set_trace as st
from math import ceil


class BRATS:
    def __init__(self, opt, phase):
        super(BRATS, self).__init__()
        random.seed(0)
        self.dataroot = opt.dataroot
        self.root = join(self.dataroot, phase)
        self.flist = []

        self.root = opt.dataroot
        ## numpy file is a list of filename paths for each slice of brain images.
        ## For fast data loading, the brain images are saved and loaded by slice-by-slice.
        self.flist = np.load(join(opt.dataroot, phase + "_flist.npy"))

        # ./datasets/FSPI/train_flist.npy

        self.N = 4
        self.nCh_out = 1
        self.nCh_in = self.N * self.nCh_out + self.N  # opt.nCh_in
        self.nY = 240
        self.nX = 240

        self.len = len(self.flist)
        self.fExp = ["D", "F", "R", "T"]  #'S'
        self.AUG = (phase == "train") and opt.AUG
        self.use_norm_std = not opt.wo_norm_std
        self.N_null = opt.N_null

        # Here, for dropout input (not used)
        self.null_N_set = [x + 1 for x in range(opt.N_null)]  # [1,2,3,4]
        self.list_for_null = []

        for i in range(self.N):
            self.list_for_null.append(self.get_null_list_for_idx(i))

    def get_info(self, opt):
        opt.nCh_in = self.nCh_in
        opt.nCh_out = self.nCh_out
        opt.nY = self.nY
        opt.nX = self.nX
        return opt

    def getBatch(self, start, end):
        nB = end - start
        end = min([end, self.len])
        start = end - nB
        batch = self.flist[start:end]

        sz_a = [end - start, self.nCh_out, self.nY, self.nX]
        sz_M = [end - start, self.nCh_out, self.nY, self.nX]

        target_class_idx = np.empty([end - start, 1], dtype=np.uint8)
        D_img = np.empty(sz_a, dtype=np.float32)
        F_img = np.empty(sz_a, dtype=np.float32)
        R_img = np.empty(sz_a, dtype=np.float32)
        T_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)

        D_mask = np.zeros(sz_M, dtype=np.float32)
        F_mask = np.zeros(sz_M, dtype=np.float32)
        R_mask = np.zeros(sz_M, dtype=np.float32)
        T_mask = np.zeros(sz_M, dtype=np.float32)

        targ_idx = random.randint(0, self.N - 1)
        tar_class_bools = [x == targ_idx for x in range(self.N)]

        for iB, aBatch in enumerate(batch):
            D_tmp = self.read_mat(aBatch + "D.mat")
            F_tmp = self.read_mat(aBatch + "F.mat")
            R_tmp = self.read_mat(aBatch + "R.mat")
            T_tmp = self.read_mat(aBatch + "T.mat")

            pad_h = 240 - D_tmp.shape[0]
            pad_w = 240 - D_tmp.shape[1]
            PT, PB = int(pad_h / 2), int((pad_h + 1) / 2)
            PL, PR = int(pad_w / 2), int((pad_w + 1) / 2)

            D_tmp = np.pad(D_tmp, ((PT, PB), (PL, PR)), "constant")
            F_tmp = np.pad(F_tmp, ((PT, PB), (PL, PR)), "constant")
            R_tmp = np.pad(R_tmp, ((PT, PB), (PL, PR)), "constant")
            T_tmp = np.pad(T_tmp, ((PT, PB), (PL, PR)), "constant")

            if self.AUG:
                if random.randint(0, 1):
                    D_tmp = np.flip(D_tmp, axis=1)
                    F_tmp = np.flip(F_tmp, axis=1)
                    R_tmp = np.flip(R_tmp, axis=1)
                    T_tmp = np.flip(T_tmp, axis=1)
                scale = random.uniform(0.9, 1.1)
                D_tmp = D_tmp * scale
                F_tmp = F_tmp * scale
                R_tmp = R_tmp * scale
                T_tmp = T_tmp * scale
            D_img[iB, :, :, :] = D_tmp
            F_img[iB, :, :, :] = F_tmp
            R_img[iB, :, :, :] = R_tmp
            T_img[iB, :, :, :] = T_tmp
            ##
            if targ_idx == 0:
                target_img[iB, :, :, :] = D_img[iB, :, :, :]
                D_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 1:
                target_img[iB, :, :, :] = F_img[iB, :, :, :]
                F_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 2:
                target_img[iB, :, :, :] = R_img[iB, :, :, :]
                R_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 3:
                target_img[iB, :, :, :] = T_img[iB, :, :, :]
                T_mask[iB, 0, :, :] = 1.0
            else:
                st()
            target_class_idx[iB] = targ_idx

        return (
            target_class_idx,
            D_img,
            F_img,
            R_img,
            T_img,
            D_mask,
            F_mask,
            R_mask,
            T_mask,
            tar_class_bools,
            target_img,
        )

    def getBatch_dir(self, start, end, _id=-1):
        nB = end - start
        end = min([end, self.len])
        start = end - nB
        batch = self.flist[start:end]

        sz_a = [end - start, self.nCh_out, self.nY, self.nX]
        sz_M = [end - start, self.nCh_out, self.nY, self.nX]

        target_class_idx = np.empty([end - start, 1], dtype=np.uint8)
        O_img = np.empty(sz_a, dtype=np.float32)
        C_img = np.empty(sz_a, dtype=np.float32)
        T_img = np.empty(sz_a, dtype=np.float32)
        F_img = np.empty(sz_a, dtype=np.float32)
        target_img = np.empty(sz_a, dtype=np.float32)

        O_mask = np.zeros(sz_M, dtype=np.float32)
        C_mask = np.zeros(sz_M, dtype=np.float32)
        T_mask = np.zeros(sz_M, dtype=np.float32)
        F_mask = np.zeros(sz_M, dtype=np.float32)

        """Thins to change for new CollaGAN DB (1/3)"""
        if _id == -1:
            targ_idx = random.randint(0, self.N - 1)
        else:
            targ_idx = _id
        tar_class_bools = [x == targ_idx for x in range(self.N)]

        for iB, aBatch in enumerate(batch):
            a_dir = aBatch
            O_tmp = self.read_mat(join(self.root, aBatch + "O.mat"))
            C_tmp = self.read_mat(join(self.root, aBatch + "C.mat"))
            T_tmp = self.read_mat(join(self.root, aBatch + "T.mat"))
            F_tmp = self.read_mat(join(self.root, aBatch + "F.mat"))

            if self.AUG:
                if random.randint(0, 1):
                    O_tmp = np.flip(O_tmp, axis=1)
                    C_tmp = np.flip(C_tmp, axis=1)
                    T_tmp = np.flip(T_tmp, axis=1)
                    F_tmp = np.flip(F_tmp, axis=1)
                scale = random.uniform(0.9, 1.1)
                O_tmp = O_tmp * scale
                C_tmp = C_tmp * scale
                T_tmp = T_tmp * scale
                F_tmp = F_tmp * scale
            O_img[iB, :, :, :] = O_tmp
            C_img[iB, :, :, :] = C_tmp
            T_img[iB, :, :, :] = T_tmp
            F_img[iB, :, :, :] = F_tmp
            ##
            if targ_idx == 0:
                target_img[iB, :, :, :] = O_img[iB, :, :, :]
                O_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 1:
                target_img[iB, :, :, :] = C_img[iB, :, :, :]
                C_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 2:
                target_img[iB, :, :, :] = T_img[iB, :, :, :]
                T_mask[iB, 0, :, :] = 1.0
            elif targ_idx == 3:
                target_img[iB, :, :, :] = F_img[iB, :, :, :]
                F_mask[iB, 0, :, :] = 1.0
            else:
                st()
            target_class_idx[iB] = targ_idx

        return (
            target_class_idx,
            O_img,
            C_img,
            T_img,
            F_img,
            O_mask,
            C_mask,
            T_mask,
            F_mask,
            tar_class_bools,
            target_img,
            a_dir,
        )

    def shuffle(self, seed=0):
        random.seed(seed)
        random.shuffle(self.flist)

    def name(self):
        return "BRATS dataset"

    def __len__(self):
        return self.len

    def get_null_list_for_idx(self, idx):
        a_list = []
        for i_null in self.null_N_set:
            tmp_a = []
            if i_null == 1:
                tmp = [bX == idx for bX in range(self.N)]
                tmp_a.append(tmp)

            elif i_null == 2:
                for i_in in range(self.N):
                    if not i_in == idx:
                        tmp = [bX in [i_in, idx] for bX in range(self.N)]
                        tmp_a.append(tmp)

            elif i_null == 3:
                for i_in in range(self.N):
                    for i2_in in range(self.N):
                        if not (i_in == i2_in or (i_in == idx or i2_in == idx)):
                            tmp = [(bX in [i_in, i2_in, idx]) for bX in range(self.N)]
                            tmp_a.append(tmp)
            elif i_null == 4:
                for i4_in in range(self.N):
                    if not (i4_in == idx):
                        tmp = [(bX == idx or (not bX == i4_in)) for bX in range(self.N)]
                        tmp_a.append(tmp)
            else:
                st()

            a_list.append(tmp_a)

        return a_list

    @staticmethod
    def read_mat(filename, var_name="img"):
        mat = sio.loadmat(filename)
        return mat[var_name]


if __name__ == "__main__":
    from options.colla_options import BaseOptions

    opt = BaseOptions().parse()
    DB_train = BRATS(opt, "train")
    st()
    idx, a, b, c, d, am, bm, cm, dm, bools, tar = DB_train.getBatch(0, 1)
    print("Return")
