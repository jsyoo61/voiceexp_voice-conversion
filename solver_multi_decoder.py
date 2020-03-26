import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import librosa
import time

from model import Encoder, Decoder, Discriminator, ACLayer, ASRLayer, SpeakerClassifier
from utils import cc
from speech_tools import sample_train_data, load_ppg, transpose_in_list, world_decompose
from tools import load_pickle, readlines, read
from fastdtw import fastdtw

from torch.autograd import Variable
from preprocess_MCD import *
from utils import Hps
from utils import Logger
from utils import DataLoader
from utils import to_var
from utils import reset_grad
from utils import multiply_grad
from utils import grad_clip
from utils import cal_acc
from utils import calculate_gradients_penalty
from utils import gen_noise
import random
from dtw import dtw

# Dummy class for debugging
# class Dummy():
#     """ Dummy class for debugging """
#     def __init__(self):
#         pass

# self = Dummy()
num_speakers = 100
batch_size = 8
train_data_dir = 'processed'
iteration = 0
self = Solver(num_speakers = num_speakers)

class Solver(object):
    def __init__(self, num_speakers = 100, log_dir='./log/'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_speakers = num_speakers
        self.dist = lambda x,y: np.linalg.norm(x-y)
        self.model_kept = []
        self.max_keep=100

        # Hyperparameters
        self.n_training_frames = 128

        self.build_model()

    def build_model(self):

        self.Encoder = cc(Encoder(label_num = self.num_speakers))
        self.Decoder = [cc(Decoder(label_num = self.num_speakers)) for i in range(self.num_speakers)]
        self.ACLayer = cc(ACLayer(label_num = self.num_speakers))
        self.Discriminator= cc(Discriminator())
        self.ASRLayer = cc(ASRLayer())
        self.SpeakerClassifier = cc(SpeakerClassifier(label_num = self.num_speakers))
        ac_betas = (0.5,0.999)
        vae_betas = (0.9,0.999)
        ac_lr = 0.00005
        vae_lr = 0.001
        dis_lr = 0.002
        clf_betas = (0.5,0.999)
        asr_betas = (0.5,0.999)
        clf_lr = 0.0002
        asr_lr = 0.00001

        decoder_parameter_list = list()
        for decoder in self.Decoder:
            decoder_parameter_list += list(decoder.parameters())
        vae_params = list(self.Encoder.parameters()) + decoder_parameter_list

        self.ac_optimizer = optim.Adam(self.ACLayer.parameters(), lr=ac_lr, betas=ac_betas)
        self.vae_optimizer = optim.Adam(vae_params, lr=vae_lr, betas=vae_betas)
        self.dis_optimizer = optim.Adam(self.Discriminator.parameters(), lr=dis_lr, betas=ac_betas)
        self.asr_optimizer = optim.Adam(self.ASRLayer.parameters(), lr=asr_lr, betas=asr_betas)
        self.clf_optimizer = optim.Adam(self.SpeakerClassifier.parameters(), lr=clf_lr, betas=clf_betas)

    def save_model(self, model_path, epoch,enc_only=True):
        all_model=dict()
        if not enc_only:
            all_model = {
                'encoder': self.Encoder.state_dict(),
                'decoder': self.Decoder.state_dict(),

                'patch_discriminator': self.PatchDiscriminator.state_dict(),
            }
        else:
            all_model['encoder'] = self.Encoder.state_dict()

            for i, decoder in enumerate(self.Decoder):
                model_name = 'decoder_' + str(i)
                all_model[model_name] = decoder.state_dict()

            all_model['aclayer'] = self.ACLayer.state_dict()

        new_model_path = os.path.join(model_path,'{}-{}'.format(model_path, epoch))
        with open(new_model_path, 'wb') as f_out:
            torch.save(all_model, f_out)
        self.model_kept.append(new_model_path)

        if len(self.model_kept) >= self.max_keep:
            os.remove(self.model_kept[0])
            self.model_kept.pop(0)

    def load_model(self, model_path, speaker_num, enc_only=True):
        speaker_num = int(speaker_num)
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])
            decoder_name = 'decoder_'+str(speaker_num)
            self.Decoder[speaker_num].load_state_dict(all_model[decoder_name])

    def load_whole_model(self,model_path,enc_only=True):
        print('load model from {}'.format(model_path))
        with open(model_path, 'rb') as f_in:
            all_model = torch.load(f_in)
            self.Encoder.load_state_dict(all_model['encoder'])

            self.Decoder.load_state_dict(all_model['decoder'])

    def set_train(self):
        self.Encoder.train()
        for decoder in self.Decoder:
            decoder.train()

    def set_eval(self,trg_speaker_num):
        self.Encoder.eval()
        self.Decoder[trg_speaker_num].eval()

    def grad_reset(self):
        self.ac_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()
        self.vae_optimizer.zero_grad()
        self.asr_optimizer.zero_grad()
        self.clf_optimizer.zero_grad()

    def test_step(self, x, c_src,c_trg,trg_speaker_num, gen=False):
        self.set_eval(trg_speaker_num)

        # Encoder
        mu_en,lv_en = self.Encoder(x,c_src)
        z = self.reparameterize(mu_en,lv_en)
        # Decoder
        xt_mu,xt_lv = self.Decoder[trg_speaker_num](mu_en, c_trg)
        x_tilde = self.reparameterize(xt_mu,xt_lv)
        if gen:
            print("add generator")
        return xt_mu.detach().cpu().numpy()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def GaussianLogDensity(self, x, mu, log_var):
        c = torch.log(torch.tensor(2*np.pi))
        var = torch.exp(log_var)
        x_mu2_over_var = ((x - mu) ** 2 ) / (var + 1e-6)
        log_prob = -0.5 * (c + log_var + x_mu2_over_var)
        return torch.mean(log_prob)

    def KLD_loss(self, mu,logvar):
        # Assume target is N(0,1)
        mu2 = torch.zeros_like(mu)
        logvar2 = torch.zeros_like(logvar)
        var = torch.exp(logvar)
        var2 = torch.exp(logvar2)
        mu_diff_sq = (mu - mu2) ** 2

        dimwise_kld = 0.5*( (logvar2 - logvar) + (var + mu_diff_sq)/(var2 + 1e-6) - 1.)
        return torch.mean(dimwise_kld)

    def CrossEnt_loss(self, logits, y_true):
        '''y_true: onehot vector'''
        loss = torch.mean(-y_true*torch.log(logits + 1e-6))
        return loss

    def clf_CrossEnt_loss(self, logits, y_true):
        '''y_true: label(indices)'''
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y_true)
        return loss

    def entropy_loss(self, logits):
        loss = torch.mean(-logits*logits.log())
        return loss

    def generate_label(self, index, batch_size):
        '''label == [index, index, ...] with len(label) == batch_size '''
        labels = [index] * int(batch_size)
        return torch.tensor(labels)

    def label2onehot(self, labels):
        # labels: [1, 1, 1, ...], with len(labels) == batch_size
        batch_size = len(labels)
        # labels = torch.tensor(labels)
        onehot = torch.zeros(batch_size, self.num_speakers)
        onehot[:, labels.long()] = 1 # labels itself are indices for the onehot vectors
        return onehot

    def encode_step(self, x,c):
        mu, logvar = self.Encoder(x,c)
        return mu, logvar

    def encode_step_cyc(self, x,c):
        mu, logvar  = self.Encoder(x,c)
        return mu, logvar

    def decode_step(self, z, c, index):
        mu, logvar = self.Decoder[index](z, c)
        return mu, logvar

    # def generator_step(self, enc, c, label):
    #     gen_mu, gen_logvar = self.Generators(enc,c)
    #     return gen_mu,gen_logvar

    def clf_step(self, x_src, label, batch_size):
        '''Get Speaker Classifier loss
        x_src - Encoder - Speaker Classifier - loss
        '''
        # Generate label
        c = self.generate_label(label,batch_size).to(device = self.device, dtype = torch.long) # [label, label, ...]
        c_src = self.generate_label(label,batch_size)
        c_src = self.label2onehot(c_src).to(device = self.device, dtype=torch.float) # label -> onehot vector

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)

        # Speaker Classifier
        logits = self.SpeakerClassifier(z)

        loss = self.clf_CrossEnt_loss(logits,c)
        return loss

    # def asr_step(self, x_src, ppg_src, label, batch_size):
    def asr_step(self, x_src, label, batch_size):
        '''Get Automatic Speech Recognizer loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        # Generate label
        c_src = self.generate_label(label, batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)

        # Automatic Speech Recognizer
        logits = self.ASRLayer(z)

        loss = self.entropy_loss(logits)
        # loss = self.CrossEnt_loss(logits, ppg_src)
        return loss

    def clf_asr_step(self, x_src,x_trg,label_src,label_trg,batch_size):
        '''Get Speaker Classifier loss & Automatic Speech Recognizer loss
        x_src - Encoder - Speaker Classifier - loss
        x_src - Encoder - Automatic Speech Recognizer - loss
        '''
        c = self.generate_label(label_src,batch_size).to(self.device, dtype = torch.long)
        c_src = self.generate_label(label_src,batch_size)
        # c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        # c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)
        # KLD = self.KLD_loss(mu_en,lv_en)

        # Speaker Classifier
        clf_logits = self.SpeakerClassifier(z)
        clf_loss = self.clf_CrossEnt_loss(clf_logits,c)

        # Automatic Speech Recognizer
        asr_logits = self.ASRLayer(z)
        asr_loss = self.entropy_loss(asr_logits)
        # loss = self.CrossEnt_loss(logits, ppg_src)

        # same_xt_mu,same_xt_lv = self.decode_step(z, c_src,label_src)
        # same_x_tilde = self.reparameterize(same_xt_mu,same_xt_lv)
        return clf_loss, asr_loss

    def vae_step(self, x_src,x_trg,label_src,label_trg,batch_size):
        '''Get Variational AutoEncoder loss
        x_src - Encoder - loss
        x_src - Encoder - Decoder - loss
        '''
        c_src = self.generate_label(label_src, batch_size)
        c_trg = self.generate_label(label_trg, batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # Decoder
        xt_mu,xt_lv = self.decode_step(z, c_trg, label_trg)
        x_tilde = self.reparameterize(xt_mu,xt_lv)

        ###loss
        KLD = self.KLD_loss(mu_en,lv_en)
        loss_rec = -self.GaussianLogDensity(x_src,xt_mu,xt_lv) # Maximize probability
        return KLD,loss_rec,x_tilde

    def cycle_step(self, x_src, x_trg, ppg_src, ppg_target, label_src, label_trg, batch_size):
        '''Get Cycle loss
        x_src - Encoder - Decoder - x_converted - Encoder - Decoder - x_reconstructed - loss
        x_src - Encoder - Decoder - x_converted - Encoder - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en, lv_en = self.encode_step(x_src, c_src)
        z = self.reparameterize(mu_en, lv_en)
        # Decoder
        convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg, label_trg)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)

        # Cycle
        # Encoder
        mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde, c_trg)
        z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)
        # Decoder
        cyc_xt_mu,cyc_xt_lv = self.decode_step(z_cyc, c_src,label_src)
        cyc_x_tilde = self.reparameterize(cyc_xt_mu,cyc_xt_lv)

        # Loss
        cyc_loss_rec = -self.GaussianLogDensity(x_src,cyc_xt_mu,cyc_xt_lv)
        cyc_KLD = self.KLD_loss(mu_en_cyc,lv_en_cyc)

        return cyc_KLD, cyc_loss_rec

    def sem_step(self, x_src,x_trg,ppg_src,ppg_target,label_src,label_trg,batch_size):
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # Decoder
        convert_xt_mu,convert_xt_lv = self.decode_step(z, c_trg,label_trg)
        convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)

        # Cycle
        # Encoder
        mu_en_cyc,lv_en_cyc = self.encode_step_cyc(convert_x_tilde,c_trg)
        z_cyc = self.reparameterize(mu_en_cyc,lv_en_cyc)

        # Mu? Z?
        # KLD_same_check = torch.mean(torch.abs(mu_en - mu_en_cyc))
        KLD_same_check = torch.mean((z - z_cyc)**2)
        return KLD_same_check

    def AC_step(self, x_src, x_trg, label_src,label_trg,batch_size):
        '''Get Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_trg = self.generate_label(label_trg,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        acc_s,src_t_label = self.ACLayer(x_src)
        acc_t,trg_t_label = self.ACLayer(x_trg)

        AC_source =  self.CrossEnt_loss(src_t_label, c_src)
        AC_target =  self.CrossEnt_loss(trg_t_label, c_trg)
        return AC_source,AC_target

    def AC_F_step(self, x_src,x_trg,ppg_src,ppg_target,label_src,label_trg,batch_size):
        '''Get Full Auxiliary Classifier loss
        x_src - Auxiliary Classifier - loss
        x_trg - Auxiliary Classifier - loss
        '''
        c_src = self.generate_label(label_src,batch_size)
        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)

        # Encoder
        mu_en,lv_en = self.encode_step(x_src,c_src)
        z = self.reparameterize(mu_en,lv_en)

        # AC layer
        acc_s,t_label = self.ACLayer(x_src)
        AC_real =  self.CrossEnt_loss(t_label, c_src)

        # Decoder step - Full
        AC_cross_list = list()
        for i in range(self.num_speakers):
            c_trg = self.generate_label(i,batch_size)
            c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)
            # Decoder
            convert_xt_mu, convert_xt_lv = self.decode_step(z, c_trg,i)
            convert_x_tilde = self.reparameterize(convert_xt_mu,convert_xt_lv)
            acc_conv_t, c_label = self.ACLayer(convert_x_tilde)
            AC_cross = self.CrossEnt_loss(c_label, c_trg)
            AC_cross_list.append(AC_cross)
        AC_cross = sum(AC_cross_list) / self.num_speakers # Mean loss

        return AC_real,AC_cross

    def patch_step(self, x, x_tilde,trg_num, batch_size, is_dis=True):
        '''Get Discriminator loss
        x - Discriminator - loss(output itself)
        x_tilde - Discriminator - loss(output itself)
        '''
        c_trg = self.generate_label(trg_num, batch_size)
        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

        D_real = self.Discriminator(x, c_trg,classify=False)
        D_fake = self.Discriminator(x_tilde, c_trg,classify=False)

        if is_dis:
            # Loss for Discriminator: D_real -> 1, D_fake -> 0
            return (-torch.mean(D_real) + torch.mean(D_fake))
        else:
            # Loss for Generator: D_fake -> 1
            return - torch.mean(D_fake)

    def MCD(self, wav, mfcc_hat, sr = 16000, frame_period = 10.0):
        # Preprocess wav
        _, _, _, _,mfcc = world_decompose(wav = wav, fs = sr, frame_period = frame_period)

        # Dynamic Time Warping(DTW)
        distance, path = fastdtw(mfcc, mfcc_hat, radius = 1000000, dist = self.dist)

        mcd = (10.0 / np.log(10)) * np.sqrt(2)* distance / len(path)
        return mcd

    def calculateMCD_manual_mfcc(self,compare_original_file_key, target_converted_source_id, compare_original_target_id,target_conv_wav_mfcc,source_wavs):
        sr = 16000
        mfcc_dim = 36
        frame_period = 10.0

        if target_converted_source_id == compare_original_target_id:
                return ""
        else:
            _,  temp = compare_original_file_key.split('_')
            pure_file_name =  temp

            _, _, _, _,sp_com_ori = world_decompose(wav = source_wavs[compare_original_file_key], fs = sr, frame_period = frame_period)
            compare_ori_wav_mfcc = sp_com_ori

            distance, path = fastdtw(compare_ori_wav_mfcc[:,1:], target_conv_wav_mfcc[:,1:], radius = 1000000, dist = self.dist)

            mcd = (10.0 / np.log(10)) * np.sqrt(2)* distance / len(path)

            resultLine = " file: " + pure_file_name + ' ' + target_converted_source_id + " -> " + compare_original_target_id + " MCD: " + str(mcd)+"\n"
            print("{} {} {} {} {}".format(model_epoch, pure_file_name, target_converted_source_id, compare_original_target_id,str(mcd)))
            return resultLine

    def train(self, batch_size, train_data_dir = 'processed', mode='train',model_iter='0'):

        # Hyperparameters
        num_mcep = 36
        ppg_dir = 'processed_ppgs_train/'

        # Hyperparameters - preprocess
        sr = 16000
        frame_period = 10.0

        speaker_list = sorted(os.listdir(train_data_dir))
i=0
j=2
src_speaker = speaker_list[i]
trg_speaker = speaker_list[j]

        if mode == 'ASR_TIMIT':
            for ep in range(1, 100 + 1):

                np.random.seed()
                for i, src_speaker in enumerate(speaker_list):
                    for j, trg_speaker in enumerate(speaker_list):

                        # Load train data, and PPG
                        # Source: A, Target: B
                        train_data_A_dir = os.path.join(train_data_dir, src_speaker, 'cache{}.p'.format(num_mcep))
                        train_data_B_dir = os.path.join(train_data_dir, trg_speaker, 'cache{}.p'.format(num_mcep))
                        ppg_A_dir = os.path.join(ppg_dir, src_speaker)
                        ppg_B_dir = os.path.join(ppg_dir, trg_speaker)

                        file_list_A, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(train_data_A_dir)
                        file_list_B, coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, log_f0s_mean_B, log_f0s_std_B = load_pickle(train_data_B_dir)
                        ppg_A = load_ppg(ppg_A_dir) # [ppg, ppg, ...], ppg.shape == (n, 144)
                        ppg_B = load_ppg(ppg_B_dir)
                        ppg_A = transpose_in_list(ppg_A) # ppg.shape == (144, n)
                        ppg_B = transpose_in_list(ppg_B)

                        dataset_A, dataset_B, ppgset_A, ppgset_B = sample_train_data(dataset_A=coded_sps_norm_A, dataset_B=coded_sps_norm_B, ppgset_A=ppg_A, ppgset_B=ppg_B, n_frames=self.n_training_frames)

                        num_data = dataset_A.shape[0]

                        dataset_A = np.expand_dims(dataset_A, axis=1)
                        dataset_A = torch.from_numpy(dataset_A).to(self.device, dtype=torch.float)
                        dataset_B = np.expand_dims(dataset_B, axis=1)
                        dataset_B = torch.from_numpy(dataset_B).to(self.device, dtype=torch.float)
                        ppgset_A = np.expand_dims(ppgset_A, axis=1)
                        ppgset_A = torch.from_numpy(ppgset_A).to(self.device, dtype=torch.float)
                        ppgset_B = np.expand_dims(ppgset_B, axis=1)
                        ppgset_B = torch.from_numpy(ppgset_B).to(self.device, dtype=torch.float)

                        print('source: %s, target: %s, num_data: %s'%(src_speaker, trg_speaker, num_data))
                        for iteration in range(4):
                            start = iteration * batch_size
                            end = (iteration + 1) * batch_size

                            x_batch_A = dataset_A[start:end]
                            x_batch_B = dataset_B[start:end]
                            ppg_batch_A = ppgset_A[start:end]
                            ppg_batch_B = ppgset_B[start:end]

                            if ((iteration+1) % 4)!=0 :
                                # Update Speaker Clssifier (CLF) module
                                self.grad_reset()
                                clf_loss_A = self.clf_step(x_batch_A, i, batch_size)
                                clf_loss_B = self.clf_step(x_batch_B, j, batch_size)
                                CLF_loss = clf_loss_A + clf_loss_B
                                loss = CLF_loss
                                loss.backward()
                                self.clf_optimizer.step()

                            if ((iteration+1) % 4)==0 :
                                # Update Automatic Speach Recognizer (ASR) module
                                self.grad_reset()
                                asr_loss_A = self.asr_step(x_batch_A, i, batch_size)
                                asr_loss_B = self.asr_step(x_batch_B, j, batch_size)
                                asr_loss = asr_loss_A + asr_loss_B
                                loss = asr_loss
                                loss.backward()
                                self.asr_optimizer.step()

                                # Update Auxiliary Classifier (AC) module
                                self.grad_reset()
                                AC_source, AC_target = self.AC_step(x_batch_A, x_batch_B, i, j, batch_size)
                                AC_t_loss = AC_source + AC_target
                                AC_t_loss.backward()
                                self.ac_optimizer.step()
                                self.grad_reset()

                                ###VAE step
                                src_KLD, src_same_loss_rec, _= self.vae_step(x_batch_A, x_batch_B, i, i, batch_size)
                                trg_KLD, trg_same_loss_rec, _= self.vae_step(x_batch_B, x_batch_A, j, j, batch_size)

                                ###AC F step
                                AC_real_src, AC_cross_src = self.AC_F_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                AC_real_trg, AC_cross_trg = self.AC_F_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                ###clf asr step
                                clf_loss_A, asr_loss_A = self.clf_asr_step(x_batch_A,x_batch_B,i,j,batch_size)
                                clf_loss_B, asr_loss_B = self.clf_asr_step(x_batch_B,x_batch_A,j,i,batch_size)
                                CLF_loss = (clf_loss_A + clf_loss_B) / 2.0
                                ASR_loss = (asr_loss_A + asr_loss_B) / 2.0

                                ###Cycle step
                                src_cyc_KLD, src_cyc_loss_rec = self.cycle_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                trg_cyc_KLD, trg_cyc_loss_rec = self.cycle_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                ###Semantic step
                                src_semloss = self.sem_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                trg_semloss = self.sem_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                AC_f_loss = (AC_real_src + AC_real_trg + AC_cross_src + AC_cross_trg) / 4.0
                                Sem_loss = (src_semloss + trg_semloss) / 2.0
                                Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD) / 2.0
                                Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec) / 2.0
                                KLD_loss = (src_KLD + trg_KLD)
                                Rec_loss = (src_same_loss_rec + trg_same_loss_rec)
                                loss = Rec_loss + KLD_loss + Cycle_KLD_loss + Cycle_rec_loss + AC_f_loss + Sem_loss-CLF_loss#+ASR_loss
                                loss.backward()
                                self.vae_optimizer.step()

                if (ep)%1==0:
                    print("Epoch : {}, Recon : {:.3f}, KLD : {:.3f}, AC t Loss : {:.3f}, AC f Loss : {:.3f}, Sem Loss : {:.3f}, Clf : {:.3f}, Asr Loss : {:.3f}"\
                        .format(ep,Rec_loss,KLD_loss,AC_t_loss,AC_cross_trg,Sem_loss,CLF_loss,ASR_loss))

                # Save model
                if (ep) % 50 ==0:
                    model_save_dir = "./VAE_all"+model_iter
                    os.makedirs(model_save_dir, exist_ok=True)
                    print("Model Save Epoch {}".format(ep))
                    self.save_model(model_save_dir, ep)

                # Validation
                if (ep) % 50 ==0:
                    wav_source_dir = "../../corpus/inset/inset_dev/"
                    validation_path_list_dir = 'filelist/in_dev.lst'
                    validation_data_dir = 'processed_validation/'
                    validation_path_list = read(validation_path_list_dir).splitlines()
                    validation_log_dir = "MCD_result_vae_epoch_"+ str(ep) +".txt"
validation_log_dir = 'test_log.txt'
                    validation_log = open(validation_log_dir, 'a')

                    validation_log.write('mode: %s\n'%(mode))
                    validation_log.write('epoch_{}\n'.format(ep))

                    for validation_path in validation_path_list:
validation_path = validation_path_list[0]

                        # Specify conversion details
                        conversion_path_sex, filename_src, filename_trg = validation_path.split()
                        src_speaker = filename_src.split('_')[0]
                        trg_speaker = filename_trg.split('_')[0]
                        label_src = speaker_list.index(src_speaker)
                        label_trg = speaker_list.index(trg_speaker)
                        wav_trg_dir = os.path.join(wav_source_dir, trg_speaker, filename_trg+'.wav')

                        # Define datapath
                        data_A_dir = os.path.join(validation_data_dir, src_speaker, '{}.p'.format(filename_src))
                        data_B_dir = os.path.join(validation_data_dir, trc_speaker, '{}.p'.format(filename_trg))
                        train_data_A_dir = os.path.join(train_data_dir, src_speaker, 'cache{}.p'.format(num_mcep))
                        train_data_B_dir = os.path.join(train_data_dir, trg_speaker, 'cache{}.p'.format(num_mcep))

                        # Load data
                        coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(os.path.join(train_data_A_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, log_f0s_mean_B, log_f0s_std_B = load_pickle(os.path.join(train_data_B_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sp, ap, f0 = load_pickle(data_A_dir)

                        # Prepare input
                        coded_sp_norm = (coded_sp - coded_sps_mean_A) / coded_sps_std_A
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = torch.from_numpy(coded_sp_norm).to(self.device, dtype=torch.float)

                        c_src = self.generate_label(label_src,1)
                        c_trg = self.generate_label(label_trg,1)
                        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
                        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

                        # Convert
                        coded_sp_converted_norm = self.test_step(coded_sp_norm, c_src, c_trg, label_trg)

                        # Post-process output
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)

                        # Additional conversions
                        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

                        # if coded_sp_converted_norm.shape[1] > len(f0):
                        #     print('shape not coherent?? file:%s'%(filename_src))
                        #     coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                        coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B

                        coded_sp_converted = coded_sp_converted.T
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                        wav_trg, _ = librosa.load(cur_wav_file_loc, sr = sr, mono = True)
                        mcd = self.MCD(wav_trg, conded_sp_converted, sr=sr, frame_period=frame_period)

                        validation_log.write(validation_path + str(mcd) + '\n')
                    print("Epoch {} : Validation Process Complete.".format(ep))
                    self.set_train()

        if mode == 'ASR_TIMIT_GAN':
            os.makedirs("./VAEGAN_all"+model_iter, exist_ok=True)
            for ep in range(1, 200 + 1):

                np.random.seed()
                for i, src_speaker in enumerate(speaker_list):
                    for j, trg_speaker in enumerate(speaker_list):

                        # Load train data, and PPG
                        # Source: A, Target: B
                        train_data_A_dir = os.path.join(train_data_dir, src_speaker, 'cache{}.p'.format(num_mcep))
                        train_data_B_dir = os.path.join(train_data_dir, trg_speaker, 'cache{}.p'.format(num_mcep))
                        ppg_A_dir = os.path.join(ppg_dir, src_speaker)
                        ppg_B_dir = os.path.join(ppg_dir, trg_speaker)

                        file_list_A, coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(train_data_A_dir)
                        file_list_B, coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, log_f0s_mean_B, log_f0s_std_B = load_pickle(train_data_B_dir)
                        ppg_A = load_ppg(ppg_A_dir) # [ppg, ppg, ...], ppg.shape == (n, 144)
                        ppg_B = load_ppg(ppg_B_dir)
                        ppg_A = transpose_in_list(ppg_A) # ppg.shape == (144, n)
                        ppg_B = transpose_in_list(ppg_B)

                        dataset_A = np.expand_dims(dataset_A, axis=1)
                        dataset_A = torch.from_numpy(dataset_A).to(self.device, dtype=torch.float)
                        dataset_B = np.expand_dims(dataset_B, axis=1)
                        dataset_B = torch.from_numpy(dataset_B).to(self.device, dtype=torch.float)
                        ppgset_A = np.expand_dims(ppgset_A, axis=1)
                        ppgset_A = torch.from_numpy(ppgset_A).to(self.device, dtype=torch.float)
                        ppgset_B = np.expand_dims(ppgset_B, axis=1)
                        ppgset_B = torch.from_numpy(ppgset_B).to(self.device, dtype=torch.float)

                        for iteration in range(81//batch_size):
                            start = iteration * batch_size
                            end = (iteration+1) * batch_size

                            x_batch_A = dataset_A[start:end]
                            x_batch_B = dataset_B[start:end]
                            ppg_batch_A = ppgset_A[start:end]
                            ppg_batch_B = ppgset_B[start:end]

                            if ((iteration+1)%5)!=0:
                                # Update Speaker Clssifier (CLF) module
                                self.grad_reset()
                                clf_loss_A = self.clf_step(x_batch_A, i, batch_size)
                                clf_loss_B = self.clf_step(x_batch_B, j, batch_size)
                                CLF_loss = clf_loss_A + clf_loss_B
                                loss = CLF_loss
                                loss.backward()
                                self.clf_optimizer.step()

                                # Update Discriminator module
                                self.grad_reset()
                                convert_KLD, convert_rec,src_to_trg_x_tilde = self.vae_step(x_batch_A,x_batch_B,i,j,batch_size)
                                trg_w_dis = self.patch_step(x_batch_B, src_to_trg_x_tilde,j, is_dis=True)
                                trg_adv_loss = trg_w_dis
                                adv_loss =  (trg_adv_loss)
                                adv_loss.backward()
                                self.dis_optimizer.step()

                                for p in self.Discriminator.parameters():
                                        p.data.clamp_(-0.01, 0.01)

                            elif ((iteration+1)%5)==0 and ep>10:
                                # Update Automatic Speach Recognizer (ASR) module
                                self.grad_reset()
                                asr_loss_A = self.asr_step(x_batch_A, i, batch_size)
                                asr_loss_B = self.asr_step(x_batch_B, j, batch_size)
                                asr_loss = asr_loss_A + asr_loss_B
                                loss = asr_loss
                                loss.backward()
                                self.asr_optimizer.step()

                                # Update Auxiliary Classifier (AC) module
                                self.grad_reset()
                                AC_source,AC_target = self.AC_step(x_batch_A,x_batch_B,i,j,batch_size)
                                AC_t_loss = AC_source+AC_target
                                AC_t_loss.backward()
                                self.ac_optimizer.step()

                                ###VAE step
                                self.grad_reset()
                                src_KLD, src_same_loss_rec, _ = self.vae_step(x_batch_A,x_batch_B,i,i,batch_size)
                                trg_KLD, trg_same_loss_rec, _ = self.vae_step(x_batch_B,x_batch_A,j,j,batch_size)

                                ###AC F step
                                AC_real_src,AC_cross_src = self.AC_F_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                AC_real_trg,AC_cross_trg = self.AC_F_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                ###clf asr step
                                clf_loss_A,asr_loss_A = self.clf_asr_step(x_batch_A,x_batch_B,i,j,batch_size)
                                clf_loss_B,asr_loss_B = self.clf_asr_step(x_batch_B,x_batch_A,j,i,batch_size)
                                CLF_loss = (clf_loss_A + clf_loss_B)/2.0
                                ASR_loss = (asr_loss_A + asr_loss_B)/2.0

                                ###Cycle step
                                src_cyc_KLD, src_cyc_loss_rec= self.cycle_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                trg_cyc_KLD, trg_cyc_loss_rec= self.cycle_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                ###Semantic step
                                src_semloss = self.sem_step(x_batch_A,x_batch_B,ppg_batch_A,ppg_batch_B,i,j,batch_size)
                                trg_semloss = self.sem_step(x_batch_B,x_batch_A,ppg_batch_B,ppg_batch_A,j,i,batch_size)

                                ###Patch step (Discriminator)
                                convert_KLD, convert_rec,src_to_trg_x_tilde = self.vae_step(x_batch_A,x_batch_B,i,j,batch_size)
                                trg_loss_adv = self.patch_step(x_batch_B, src_to_trg_x_tilde,j, is_dis=False)

                                AC_f_loss = (AC_real_src+AC_real_trg+AC_cross_src+AC_cross_trg)/4.0
                                Sem_loss = (src_semloss+trg_semloss)/2.0
                                Cycle_KLD_loss = (src_cyc_KLD + trg_cyc_KLD)/2.0
                                Cycle_rec_loss = (src_cyc_loss_rec + trg_cyc_loss_rec)/2.0
                                KLD_loss = (src_KLD+trg_KLD)
                                Rec_loss = (src_same_loss_rec+trg_same_loss_rec)
                                loss = Rec_loss + KLD_loss + Cycle_KLD_loss + Cycle_rec_loss + AC_f_loss + Sem_loss - CLF_loss + trg_loss_adv #+ASR_loss
                                loss.backward()
                                self.vae_optimizer.step()

                if ep>10:
                    print("Epoch : {}, Recon Loss : {:.3f},  KLD Loss : {:.3f}, Dis Loss : {:.3f},  GEN Loss : {:.3f}, AC t Loss : {:.3f}, AC f Loss : {:.3f}".format(ep,Rec_loss,KLD_loss,adv_loss,trg_loss_adv,AC_t_loss,AC_cross_trg))
                else:
                    print("Epoch : {} Dis Loss : {}".format(ep,adv_loss))

                # Save model
                if (ep) % 50 ==0:
                    print("Model Save Epoch {}".format(ep))
                    self.save_model("VAEGAN_all"+model_iter, ep)

                # Validation
                if (ep) % 50 ==0:
validation_log_dir = 'test_log.txt'
                    wav_source_dir = "../../corpus/inset/inset_dev/"
                    validation_path_list_dir = 'filelist/in_dev.lst'
                    validation_data_dir = 'processed_validation/'
                    validation_path_list = read(validation_path_list_dir).splitlines()
                    validation_log_dir = "MCD_result_vae_epoch_"+ str(ep) +".txt"
                    validation_log = open(validation_log_dir, 'a')

                    validation_log.write('epoch_{}\n'.format(ep))

                    for validation_path in validation_path_list:
validation_path = validation_path_list[0]

                        # Specify conversion details
                        conversion_path_sex, filename_src, filename_trg = validation_path.split()
                        src_speaker = filename_src.split('_')[0]
                        trg_speaker = filename_trg.split('_')[0]
                        label_src = speaker_list.index(src_speaker)
                        label_trg = speaker_list.index(trg_speaker)
                        wav_trg_dir = os.path.join(wav_source_dir, trg_speaker, filename_trg+'.wav')

                        # Define datapath
                        data_A_dir = os.path.join(validation_data_dir, src_speaker, '{}.p'.format(filename_src))
                        data_B_dir = os.path.join(validation_data_dir, trc_speaker, '{}.p'.format(filename_trg))
                        train_data_A_dir = os.path.join(train_data_dir, src_speaker, 'cache{}.p'.format(num_mcep))
                        train_data_B_dir = os.path.join(train_data_dir, trg_speaker, 'cache{}.p'.format(num_mcep))

                        # Load data
                        coded_sps_norm_A, coded_sps_mean_A, coded_sps_std_A, log_f0s_mean_A, log_f0s_std_A = load_pickle(os.path.join(train_data_A_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sps_norm_B, coded_sps_mean_B, coded_sps_std_B, log_f0s_mean_B, log_f0s_std_B = load_pickle(os.path.join(train_data_B_dir, 'cache{}.p'.format(num_mcep)))
                        coded_sp, ap, f0 = load_pickle(data_A_dir)

                        # Prepare input
                        coded_sp_norm = (coded_sp - coded_sps_mean_A) / coded_sps_std_A
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = np.expand_dims(coded_sp_norm, axis=0)
                        coded_sp_norm = torch.from_numpy(coded_sp_norm).to(self.device, dtype=torch.float)

                        c_src = self.generate_label(label_src,1)
                        c_trg = self.generate_label(label_trg,1)
                        c_src = self.label2onehot(c_src).to(self.device, dtype=torch.float)
                        c_trg = self.label2onehot(c_trg).to(self.device, dtype=torch.float)

                        # Convert
                        coded_sp_converted_norm = self.test_step(coded_sp_norm, c_src, c_trg, label_trg)

                        # Post-process output
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)
                        coded_sp_converted_norm = np.squeeze(coded_sp_converted_norm, axis=0)

                        # Additional conversions
                        f0_converted = pitch_conversion(f0=f0, mean_log_src=log_f0s_mean_A, std_log_src=log_f0s_std_A, mean_log_target=log_f0s_mean_B, std_log_target=log_f0s_std_B)

                        # if coded_sp_converted_norm.shape[1] > len(f0):
                        #     print('shape not coherent?? file:%s'%(filename_src))
                        #     coded_sp_converted_norm = coded_sp_converted_norm[:, :-1]
                        coded_sp_converted = coded_sp_converted_norm * coded_sps_std_B + coded_sps_mean_B

                        coded_sp_converted = coded_sp_converted.T
                        coded_sp_converted = np.ascontiguousarray(coded_sp_converted)

                        wav_trg, _ = librosa.load(cur_wav_file_loc, sr = sr, mono = True)
                        mcd = self.MCD(wav_trg, conded_sp_converted, sr=sr, frame_period=frame_period)

                        validation_log.write(validation_path + str(mcd) + '\n')
                    print("Epoch {} : Validation Process Complete.".format(ep))
                    self.set_train()

# x_src = x_batch_A
# x_trg = x_batch_B
# ppg_src = ppg_batch_A
# ppg_trg = ppg_batch_B
# label_src = i
# label_trg = j

# ppg_A = load_pickle('processed_ppgs_train/p226/ppgs_train.p')
# for ppg in ppg_A:
#     print(ppg.shape)
# dataset_A.shape
# dataset_B.shape
# ppgset_A.shape
# for ppg in ppgset_A:
#     print(ppg.shape)
# len(coded_sps_norm_A)
# len(coded_sps_norm_B)
# len(ppg_A)
# len(ppg_B)
# trg_speaker
# train_data_B_dir
# coded_sps_norm_A[0].shape
# for c, d in zip(coded_sps_norm_A, ppg_A):
#     print(c.shape, d.shape)
# for c, d in zip(coded_sps_norm_B, ppg_B):
#     print(c.shape, d.shape)
#
# ppg_A
# len(ppg_B)
# ppg_A[0].shape
# ppg_A[1].shape
# file_dir = 'G:\\vctk\\inset_train\\p288'
# file_list = sorted(os.listdir(file_dir))
# file_list
# ppg_B = list()
# for file in file_list:
#     file_dir_1 = os.path.join(file_dir, file)
#     ppg = load_pickle(file_dir_1)
#     ppg_B.append(ppg)
#
# for ppg in ppg_B:
#     print(ppg.shape)
# a

c.shape
c
c.dtype
c.byte()
c.long()

c.to(dtype = torch.LongTensor)
c.to(torch.LongTensor)
c.dtype
len(c)
torch.tensor(c)
o = torch.zeros(8, 100)
o[:, :10] = 1
o[:, c]
o[:, c.byte()]
o[:, c.long()]
o.shape
onehot = self.label2onehot(c, 8)
onehot.shape
onehot[0]
