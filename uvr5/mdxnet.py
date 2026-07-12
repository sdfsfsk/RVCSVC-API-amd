import os
import logging
import traceback
logger = logging.getLogger(__name__)

import librosa,ffmpeg
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

_use_dml = os.environ.get("RVC_USE_DML", "0") == "1"
if _use_dml:
    try:
        import torch_directml
        cpu = torch_directml.device(torch_directml.default_device())
        is_half = False
        print(f"[AMD] mdxnet 使用 DirectML 设备: {cpu}")
    except ImportError:
        print("[AMD] torch_directml 未安装，mdxnet 回退到 CPU")
        cpu = 'cpu'
        is_half = False
else:
    cpu = 'cuda' if torch.cuda.is_available() else 'cpu'
    is_half = cpu == torch.device("cuda")

class ConvTDFNetTrim:
    def __init__(
        self, device, model_name, target_name, L, dim_f, dim_t, n_fft, hop=1024
    ):
        super(ConvTDFNetTrim, self).__init__()

        self.dim_f = dim_f
        self.dim_t = 2**dim_t
        self.n_fft = n_fft
        self.hop = hop
        self.n_bins = self.n_fft // 2 + 1
        self.chunk_size = hop * (self.dim_t - 1)
        self.window = torch.hann_window(window_length=self.n_fft, periodic=True)
        self.target_name = target_name
        self.blender = "blender" in model_name

        self.dim_c = 4
        out_c = self.dim_c * 4 if target_name == "*" else self.dim_c
        self.freq_pad = torch.zeros(
            [1, out_c, self.n_bins - self.dim_f, self.dim_t]
        )

        self.n = L // 2

    def stft(self, x):
        orig_device = x.device
        x_cpu = x.cpu()
        window_cpu = self.window.to(x_cpu.device)
        x_cpu = x_cpu.reshape([-1, self.chunk_size])
        x_cpu = torch.stft(
            x_cpu,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=window_cpu,
            center=True,
            return_complex=True,
        )
        x_cpu = torch.view_as_real(x_cpu)
        x_cpu = x_cpu.permute([0, 3, 1, 2])
        x_cpu = x_cpu.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape(
            [-1, self.dim_c, self.n_bins, self.dim_t]
        )
        result = x_cpu[:, :, : self.dim_f]
        return result.to(orig_device)

    def istft(self, x, freq_pad=None):
        orig_device = x.device
        x_cpu = x.cpu()
        if freq_pad is None:
            freq_pad_cpu = self.freq_pad.repeat([x_cpu.shape[0], 1, 1, 1])
        else:
            freq_pad_cpu = freq_pad.cpu()
        x_cpu = torch.cat([x_cpu, freq_pad_cpu], -2)
        c = 4 * 2 if self.target_name == "*" else 2
        x_cpu = x_cpu.reshape([-1, c, 2, self.n_bins, self.dim_t]).reshape(
            [-1, 2, self.n_bins, self.dim_t]
        )
        x_cpu = x_cpu.permute([0, 2, 3, 1])
        x_cpu = x_cpu.contiguous()
        x_cpu = torch.view_as_complex(x_cpu)
        window_cpu = self.window.to(x_cpu.device)
        if is_half:
            window_cpu = window_cpu.half()
        x_cpu = torch.istft(
            x_cpu, n_fft=self.n_fft, hop_length=self.hop, window=window_cpu, center=True
        )
        result = x_cpu.reshape([-1, c, self.chunk_size])
        return result.to(orig_device)


def get_models(device, dim_f, dim_t, n_fft):
    return ConvTDFNetTrim(
        device=device,
        model_name="Conv-TDF",
        target_name="UVR_MDXNET_KARA_2",
        L=11,
        dim_f=dim_f,
        dim_t=dim_t,
        n_fft=n_fft,
    )


class Predictor:
    def __init__(self, args):
        import onnxruntime as ort

        logger.info(ort.get_available_providers())
        self.args = args
        self.model_ = get_models(
            device=cpu, dim_f=args.dim_f, dim_t=args.dim_t, n_fft=args.n_fft
        )
        providers = []
        if _use_dml:
            providers = ["DmlExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CUDAExecutionProvider", "DmlExecutionProvider", "CPUExecutionProvider"]
        self.model = ort.InferenceSession(
            args.onnx,
            providers=providers,
        )
        logger.info("ONNX load done")

    def demix(self, mix):
        samples = mix.shape[-1]
        margin = self.args.margin
        chunk_size = self.args.chunks * 44100
        assert not margin == 0, "margin cannot be zero!"
        if margin > chunk_size:
            margin = chunk_size

        segmented_mix = {}

        if self.args.chunks == 0 or samples < chunk_size:
            chunk_size = samples

        counter = -1
        for skip in range(0, samples, chunk_size):
            counter += 1

            s_margin = 0 if counter == 0 else margin
            end = min(skip + chunk_size + margin, samples)

            start = skip - s_margin

            segmented_mix[skip] = mix[:, start:end].copy()
            if end == samples:
                break

        sources = self.demix_base(segmented_mix, margin_size=margin)
        """
        mix:(2,big_sample)
        segmented_mix:offset->(2,small_sample)
        sources:(1,2,big_sample)
        """
        return sources

    def demix_base(self, mixes, margin_size):
        chunked_sources = []
        progress_bar = tqdm(total=len(mixes))
        progress_bar.set_description("Processing")
        for mix in mixes:
            cmix = mixes[mix]
            sources = []
            n_sample = cmix.shape[1]
            model = self.model_
            trim = model.n_fft // 2
            gen_size = model.chunk_size - 2 * trim
            pad = gen_size - n_sample % gen_size
            mix_p = np.concatenate(
                (np.zeros((2, trim)), cmix, np.zeros((2, pad)), np.zeros((2, trim))), 1
            )
            mix_waves = []
            i = 0
            while i < n_sample + pad:
                waves = np.array(mix_p[:, i : i + model.chunk_size])
                mix_waves.append(waves)
                i += gen_size
            dtype = torch.float16 if is_half else torch.float32
            mix_waves = torch.tensor(mix_waves, dtype=dtype).to(cpu)
            with torch.no_grad():
                _ort = self.model
                spek = model.stft(mix_waves)
                if self.args.denoise:
                    spec_pred = (
                        -_ort.run(None, {"input": -spek.cpu().numpy()})[0] * 0.5
                        + _ort.run(None, {"input": spek.cpu().numpy()})[0] * 0.5
                    )
                    tar_waves = model.istft(torch.tensor(spec_pred, dtype=dtype).to(cpu))
                else:
                    tar_waves = model.istft(
                        torch.tensor(_ort.run(None, {"input": spek.cpu().numpy()})[0], dtype=dtype).to(cpu)
                    )
                tar_signal = (
                    tar_waves[:, :, trim:-trim]
                    .transpose(0, 1)
                    .reshape(2, -1)
                    .cpu()  # 先移到 CPU
                    .numpy()[:, :-pad]  # 然后转为 NumPy 数组
                )

                start = 0 if mix == 0 else margin_size
                end = None if mix == list(mixes.keys())[::-1][0] else -margin_size
                if margin_size == 0:
                    end = None
                sources.append(tar_signal[:, start:end])

                progress_bar.update(1)

            chunked_sources.append(sources)
        _sources = np.concatenate(chunked_sources, axis=-1)
        # del self.model
        progress_bar.close()
        return _sources

    def prediction(self, m, vocal_root, others_root, format):
        os.makedirs(vocal_root, exist_ok=True)
        os.makedirs(others_root, exist_ok=True)
        basename = os.path.basename(m)
        mix, rate = librosa.load(m, mono=False, sr=44100)
        if mix.ndim == 1:
            mix = np.asfortranarray([mix, mix])
        mix = mix.T
        sources = self.demix(mix.T)
        opt = sources[0].T
        if format in ["wav", "flac"]:
            sf.write(
                "%s/%s_main_vocal.%s" % (vocal_root, basename, format), mix - opt, rate
            )
            sf.write("%s/%s_others.%s" % (others_root, basename, format), opt, rate)
        else:
            path_vocal = "%s/%s_main_vocal.wav" % (vocal_root, basename)
            path_other = "%s/%s_others.wav" % (others_root, basename)
            sf.write(path_vocal, mix - opt, rate)
            sf.write(path_other, opt, rate)
            opt_path_vocal = path_vocal[:-4] + ".%s" % format
            opt_path_other = path_other[:-4] + ".%s" % format
            if os.path.exists(path_vocal):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y" % (path_vocal, opt_path_vocal)
                )
                if os.path.exists(opt_path_vocal):
                    try:
                        os.remove(path_vocal)
                    except:
                        pass
            if os.path.exists(path_other):
                os.system(
                    "ffmpeg -i %s -vn %s -q:a 2 -y" % (path_other, opt_path_other)
                )
                if os.path.exists(opt_path_other):
                    try:
                        os.remove(path_other)
                    except:
                        pass


class MDXNetDereverb:
    def __init__(self, chunks,model_path):
        self.onnx = model_path
        self.shifts = 10  # 'Predict with randomised equivariant stabilisation'
        self.mixing = "min_mag"  # ['default','min_mag','max_mag']
        self.chunks = chunks
        self.margin = 44100
        self.dim_t = 9
        self.dim_f = 3072
        self.n_fft = 6144
        self.denoise = True
        self.pred = Predictor(self)
        self.device = cpu

    def _path_audio_(self, input, others_root, vocal_root, format, is_hp3=False):
        need_reformat = 1
        done = 0
        try:
            info = ffmpeg.probe(input, cmd="ffprobe")
            if (
                info["streams"][0]["channels"] == 2
                and info["streams"][0]["sample_rate"] == "44100"
            ):
                need_reformat = 0
                self.pred.prediction(input, vocal_root, others_root, format)
                done = 1
        except:
            need_reformat = 1
            traceback.print_exc()
        if need_reformat == 1:
            tmp_path = "%s/%s.reformatted.wav" % (
                os.path.join(os.environ["TEMP"]),
                os.path.basename(input),
            )
            os.system(
                f'ffmpeg -i "{input}" -vn -acodec pcm_s16le -ac 2 -ar 44100 "{tmp_path}" -y'
            )
            input = tmp_path
        try:
            if done == 0:
                self.pred.prediction(input, vocal_root, others_root, format)
            print("%s->Success" % (os.path.basename(input)))

        except:
            print(
                "%s->%s" % (os.path.basename(input), traceback.format_exc())
            )

        
