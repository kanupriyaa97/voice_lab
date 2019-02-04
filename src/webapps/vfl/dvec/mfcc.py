import numpy as np
import numpy.fft as fft


def _fs2mel(f):
    return 1125. * np.log(1. + f / 700.)

def _mel2fs(m):
    return 700. * (np.exp(m / 1125.) - 1.)

class MFCC(object):
    def __init__(self, alpha=0.97, srate=8000, frate=100, tlen=25.,
                 nfft=512, lowerf=80., upperf=7200., nfilt=40, ncep=13):
        # setting
        self.alpha = alpha
        self.srate = srate
        self.wlen = int(np.round(srate * tlen / 1000.))
        self.stride = int(np.round(srate / frate))
        self.lowerf = lowerf
        self.upperf = upperf
        self.nfft = nfft
        self.nfilt = nfilt
        self.ncep = ncep
        self.win = np.hamming(self.wlen)

        self.mel_fbank = self._create_mel_filterbank()
        self.dctmat = self._create_dctmat()

    def _create_mel_filterbank(self):
        minmel = _fs2mel(self.lowerf)
        maxmel = _fs2mel(self.upperf)
        mel_bw = (maxmel - minmel) / (self.nfilt + 1.)
        mels = minmel + np.arange(self.nfilt + 2) * mel_bw
        freqs = _mel2fs(mels)
        indices = np.round(self.nfft * freqs / self.srate)
        #print indices
        mel_fbank = np.zeros(shape=(self.nfilt, self.nfft/2 + 1))
        for i in xrange(self.nfilt):
            start = int(indices[i])
            middle = int(indices[i+1])
            end = int(indices[i+2])
            if not end - start > 0:
               print i
               height = 0.
            else:
               height = 2. / (end - start) * self.nfft / self.srate
            mel_fbank[i, start:middle] = height * (np.arange(start, middle) - start) / float(middle - start)
            mel_fbank[i, middle:end] = height * (end - 1. - np.arange(middle, end)) / float(end - middle)
        return mel_fbank

    def _create_dctmat(self):
        dctmat = np.empty((self.ncep, self.nfilt), 'double')
        for i in xrange(self.ncep):
            freq = np.pi * float(i) / self.nfilt
            dctmat[i] = np.cos(freq * np.arange(0.5, float(self.nfilt) + 0.5, 1.0, 'double'))
        dctmat[:,0] = dctmat[:,0] * 0.5
        return dctmat

    def _pre_emphasis(self, sig):
        sig = np.insert(sig, 0, 0)
        return sig[1:] - self.alpha * sig[:-1]

    def _frame2fspec(self, frame):
        # hamming window & fft
        fft_coefs = np.fft.rfft(frame * self.win, self.nfft)
        return np.log(np.absolute(fft_coefs).clip(1e-5, np.inf))

    def _frame2logspec(self, frame):
        # hamming window & fft
        fft_coefs = np.fft.rfft(frame * self.win, self.nfft)
        # power spectrum
        pow_spec = np.square(np.absolute(fft_coefs))
        return np.log(np.dot(pow_spec, self.mel_fbank.T).clip(1e-5, np.inf))

    def _frame2melspec(self, frame):
        logspec = self._frame2logspec(frame)
        return np.dot(logspec, self.dctmat.T) / self.nfilt

    def compute_fspec(self, sig):
        ## 1 - pre emphasis
        emph_sig = self._pre_emphasis(sig)
        ##
        s_pt = 0
        e_pt = s_pt + self.wlen
        length = len(emph_sig)
        frame_n = int((length - self.wlen) / self.stride) + 1
        fspec = np.empty(shape=[frame_n, self.nfft/2 + 1])
        while not e_pt > length:
            frame = emph_sig[s_pt:e_pt]
            fspec[int(s_pt/self.stride), :] = self._frame2fspec(frame)
            s_pt += self.stride
            e_pt = s_pt + self.wlen
        return fspec
        
    def compute_logspec(self, sig):
        ## 1 - pre emphasis
        emph_sig = self._pre_emphasis(sig)
        ##
        s_pt = 0
        e_pt = s_pt + self.wlen
        length = len(emph_sig)
        frame_n = int((length - self.wlen) / self.stride) + 1
        logspec = np.empty(shape=[frame_n, self.nfilt])
        while not e_pt > length:
            frame = emph_sig[s_pt:e_pt]
            logspec[int(s_pt/self.stride), :] = self._frame2logspec(frame)
            s_pt += self.stride
            e_pt = s_pt + self.wlen
        return logspec

    def compute_melspec(self, sig):
        ## 1 - pre emphasis
        emph_sig = self._pre_emphasis(sig)
        ##
        s_pt = 0
        e_pt = s_pt + self.wlen
        length = len(emph_sig)
        frame_n = int((length - self.wlen) / self.stride) + 1
        melspec = np.empty(shape=[frame_n, self.ncep])
        while not e_pt > length:
            frame = emph_sig[s_pt:e_pt]
            melspec[int(s_pt/self.stride), :] = self._frame2melspec(frame)
            s_pt += self.stride
            e_pt = s_pt + self.wlen
        return melspec
