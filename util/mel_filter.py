import torch
import torch.fft as fft
from librosa.filters import mel as librosa_mel

def pad_center(x, target_size):
    current_size = x.shape[-1]
    if current_size >= target_size:
        return x
    
    pad_size = target_size - current_size
    pad_left = pad_size // 2
    pad_right = pad_size - pad_left  # 处理奇数情况
    padded = torch.nn.functional.pad(x, (pad_left, pad_right, 0, 0))
    return padded

def lowpass_fourier_filter_1d(x, threshold, scale = 1, dim = (-1), return_freq = False):
    dtype = x.dtype
    x = x.type(torch.float32)
    x_freq = fft.fft(x, dim=dim)
    x_freq = fft.fftshift(x_freq, dim=dim)
        
    # Create mask to filter high frequencies
    H, W = x_freq.shape[-2:]
    mask = torch.zeros((H, W), device=x.device)
    center = W // 2
    start, end = max(0, center - threshold), min(W, center + threshold)
    mask[:, start:end] = scale
    x_freq = x_freq * mask

    # Only get the real part and imaginary part
    if return_freq:
        x_freq = x_freq[...,start:end]
        return x_freq

    # get the reconstructed mel
    x_freq = fft.ifftshift(x_freq, dim=dim)
    x_filtered = fft.ifft(x_freq, dim=dim).real
    x_filtered = x_filtered.type(dtype)
    return x_filtered


def Reconstruct_from_freq(x_freq, x_length,dim = -1):
    x_freq_real = x_freq[..., ::2]  # even indices for real part
    x_freq_imag = x_freq[..., 1::2]  # odd indices for imaginary part
    x_freq = torch.complex(x_freq_real, x_freq_imag)
    x_freq = pad_center(x_freq, x_length)
    x_freq = fft.ifftshift(x_freq, dim=dim)
    x_filtered = fft.ifft(x_freq, dim=dim).real
    return x_filtered


def create_hidden_weights(batch_size, length, hidden_dim, mode = "Linear", mask_size = None):
    if mode == "Linear":
        num_pairs = (hidden_dim + 1) // 2  # 使用整除，处理维度为奇数的情况
        scale = 4
        # pair_weights1 = torch.sigmoid(torch.linspace(-scale, scale, num_pairs))
        # pair_weights2 = torch.sigmoid(torch.linspace(scale, -scale, num_pairs))

        pair_weights1 = torch.linspace(0.1, 1, num_pairs)
        pair_weights2 = torch.linspace(1, 0.1, num_pairs)

        weights = torch.concat([pair_weights1, pair_weights2])
        weights = weights.view(1, 1, -1)
        weights = weights.expand(batch_size, length, hidden_dim)
        weights =  weights/weights.mean()
    elif mode == "Mask":
        assert mask_size != None
        weights = torch.zeros(batch_size, length, hidden_dim)
        start_idx = (hidden_dim - mask_size) // 2
        end_idx = start_idx + mask_size
        weights[:, :, start_idx:end_idx] = 1
    else:
        raise NotImplementedError(f"{mode} weighted Loss is not supported.")
    
    return weights


def extract_batch_mel(waveform, 
                      cut_audio_duration, 
                      sampling_rate, 
                      hop_length,
                      maximum_amplitude,
                      filter_length,
                      n_mel,
                      mel_fmin,
                      mel_fmax,
                      win_length):
    target_mel_length = int(cut_audio_duration * sampling_rate / hop_length)
    waveform = waveform - torch.mean(waveform, dim=1, keepdim=True)
    waveform = waveform / (torch.max(torch.abs(waveform), dim=1).values.unsqueeze(1) + 1e-8)
    waveform = waveform * maximum_amplitude

    waveform = waveform.unsqueeze(0)
    waveform = torch.nn.functional.pad(
        waveform,
        ( int((filter_length - hop_length) / 2), int((filter_length - hop_length) / 2), 0, 0),
        mode="reflect",)
    waveform = waveform.squeeze(0)


    mel_basis = librosa_mel(
        sr=sampling_rate,
        n_fft=filter_length,
        n_mels=n_mel,
        fmin=mel_fmin,
        fmax=mel_fmax,
    )
    mel_basis = torch.from_numpy(mel_basis).float().to(waveform.device)
    hann_window = torch.hann_window(win_length).to(waveform.device)


    def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
        return torch.log(torch.clamp(x, min=clip_val) * C)
    stft_spec = torch.stft(
        waveform,
        filter_length,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window,
        center=False,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=True,
    )
    stft_spec = torch.abs(stft_spec)
    mel = dynamic_range_compression_torch( torch.matmul(mel_basis, stft_spec) )



    def pad_spec(cur_log_mel_spec):
        n_frames = cur_log_mel_spec.shape[-2]
        p = target_mel_length - n_frames
        # cut and pad
        if p > 0:
            m = torch.nn.ConstantPad2d((0, 0, 0, p), value=-11)
            cur_log_mel_spec = m(cur_log_mel_spec)
        elif p < 0:
            cur_log_mel_spec = cur_log_mel_spec[..., 0 : target_mel_length, :]

        if cur_log_mel_spec.size(-1) % 2 != 0:
            cur_log_mel_spec = cur_log_mel_spec[..., :-1]
        return cur_log_mel_spec
    log_mel_spec, stft = mel.to(torch.float).transpose(1,2), stft_spec.to(torch.float).transpose(1,2)
    log_mel_spec, stft = pad_spec(log_mel_spec), pad_spec(stft)

    return waveform, log_mel_spec

