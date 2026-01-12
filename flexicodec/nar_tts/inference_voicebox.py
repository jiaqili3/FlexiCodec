#!/usr/bin/env python3
# Copyright (c) 2025 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch
import torchaudio
import soundfile as sf
from functools import partial
import torch.nn.functional as F
from typing import Dict, Optional
import sys
sys.path.append(f'{os.path.dirname(__file__)}/../..')
from flexicodec.nar_tts.modeling_voicebox import VoiceboxWrapper
from flexicodec.nar_tts.vocoder_model import get_vocos_model_spectrogram, mel_to_wav_vocos

from flexicodec.feature_extractors import FBankGen

# Global feature extractor for dualcodec
feature_extractor_for_dualcodec = FBankGen(sr=16000)

# Global model cache
_model_cache = None
_vocoder_cache = None

def prepare_voicebox_model(
    checkpoint_path: str,
    voicebox_config: Optional[Dict] = None,
    device: Optional[str] = None,
    n_timesteps: int = 15,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
):
    """
    Prepare and load the VoiceboxWrapper model and vocoder for inference.
    
    Args:
        checkpoint_path: Path to the model checkpoint (.pt or .safetensors)
        voicebox_config: Optional VoiceBox model configuration dict. If None, uses default config.
        device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detection)
        n_timesteps: Number of diffusion timesteps (default: 15)
        cfg: Classifier-free guidance scale (default: 2.0)
        rescale_cfg: Rescaling factor for CFG (default: 0.75)
    
    Returns:
        dict: Dictionary containing 'model', 'vocoder_decode_func', and inference parameters
    """
    global _model_cache, _vocoder_cache
    
    # Default VoiceBox config
    if voicebox_config is None:
        voicebox_config = {
            'mel_dim': 128,
            'hidden_size': 1024,
            'num_layers': 16,
            'num_heads': 16,
            'cfg_scale': 0.2,
            'use_cond_code': True,
            'cond_codebook_size': 32768,
            'cond_scale_factor': 4,
            'cond_dim': 1024,
            'sigma': 1e-5,
            'time_scheduler': "cos"
        }
    
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    
    print(f"Loading VoiceboxWrapper model...")
    model = VoiceboxWrapper(voicebox_config=voicebox_config)
    
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if checkpoint_path.endswith('.safetensors'):
        import safetensors.torch
        state_dict = safetensors.torch.load_file(checkpoint_path)
    else:
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        state_dict = ckpt.get('model', ckpt.get('state_dict', ckpt))
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model = model.to(device)
    
    # Load vocoder
    print(f"Loading vocoder on {device}...")
    vocoder_decode_func, _ = load_vocoder(device)
    
    return {
        'model': model,
        'vocoder_decode_func': vocoder_decode_func,
        'device': device,
        'n_timesteps': n_timesteps,
        'cfg': cfg,
        'rescale_cfg': rescale_cfg,
    }


def infer_voicebox_tts(
    model_dict: Dict,
    audio_tokens: torch.Tensor,
    length_ids: Optional[torch.Tensor] = None,
    prompt_audio: Optional[torch.Tensor] = None,
    prompt_audio_path: Optional[str] = None,
    framerate: Optional[float] = None,
) -> tuple:
    """
    Perform Voicebox-based NAR TTS inference.
    
    Args:
        model_dict: Dictionary returned from prepare_voicebox_model()
        audio_tokens: [T] audio token indices (semantic codes)
        length_ids: [T] length class indices (token durations)
        prompt_audio: Optional prompt audio tensor [1, T_audio]. If None, uses placeholder.
        prompt_audio_path: Optional path to prompt audio file for caching features
        framerate: Optional frame rate. If None, uses default framerate (1.0).
    
    Returns:
        tuple: (output_audio_tensor, sample_rate)
            - output_audio_tensor: Generated audio tensor [T] or [1, T]
            - sample_rate: Sample rate of output audio (16000 or 24000 Hz)
    """
    model = model_dict['model']
    vocoder_decode_func = model_dict['vocoder_decode_func']
    device = model_dict['device']
    
    # Set default framerate if not provided
    if framerate is None:
        framerate = 1.0
        print(f"Framerate not given. Using default framerate: {framerate}.")
    
    # Prepare audio tokens
    semantic_codes = audio_tokens
    if semantic_codes.dim() == 1:
        semantic_codes = semantic_codes.unsqueeze(0)  # [1, T]
    semantic_codes = semantic_codes.to(device)
    
    # Convert length_ids to token_lengths if provided
    if length_ids is not None:
        token_lengths = length_ids
        if token_lengths.dim() == 1:
            token_lengths = token_lengths.unsqueeze(0)  # [1, T]
        token_lengths = token_lengths.to(device)
    else:
        token_lengths = torch.ones(1, semantic_codes.size(1), dtype=torch.long, device=device)
    
    # Expand generated tokens using duration classes (token_lengths)
    expanded_gen_tokens = semantic_codes
    if length_ids is not None:
        expanded_gen_tokens = torch.repeat_interleave(semantic_codes[0], token_lengths[0]).unsqueeze(0)
    
    # Load prompt audio from path if provided
    if prompt_audio_path is not None:
        prompt_audio, prompt_sr = torchaudio.load(prompt_audio_path)
        if prompt_sr != 16000:
            prompt_audio = torchaudio.transforms.Resample(prompt_sr, 16000)(prompt_audio)
    else:
        assert prompt_audio is not None, "prompt_audio must be provided"
    # Ensure prompt audio is mono and properly shaped
    if prompt_audio.dim() == 1:
        prompt_audio = prompt_audio.unsqueeze(0)
    if prompt_audio.shape[0] > 1:
        prompt_audio = prompt_audio.mean(dim=0, keepdim=True)
    prompt_audio = prompt_audio.to(device)
    
    # Extract features for prompt audio
    prompt_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(prompt_audio.cpu())
    prompt_mel_features = prompt_mel_features.to(device)
    prompt_x_lens = torch.tensor([prompt_mel_features.shape[1]], dtype=torch.long, device=device)
    
    # Set merging threshold on model
    model.dualcodec_model.similarity_threshold = framerate
    
    # Extract semantic codes from prompt
    prompt_output = model._extract_dualcodec_features(
        prompt_audio, mel=prompt_mel_features, x_lens=prompt_x_lens, manual_threshold=framerate
    )
    prompt_cond_codes = prompt_output['semantic_codes_aggregated'].squeeze(1)  # [1, T_prompt]
    prompt_token_lengths = prompt_output.get('token_lengths')
    
    # Expand prompt tokens if token_lengths are available
    expanded_prompt_tokens = prompt_cond_codes
    if prompt_token_lengths is not None and prompt_cond_codes.shape[1] > 0:
        expanded_prompt_tokens = torch.repeat_interleave(prompt_cond_codes[0], prompt_token_lengths[0]).unsqueeze(0)
    
    # Extract prompt mel features
    prompt_mel = model._extract_mel_features(prompt_audio)
    
    # Concatenate expanded prompt and expanded generated codes
    cond_codes = torch.cat([expanded_prompt_tokens, expanded_gen_tokens], dim=1)
    
    # Get conditioning features
    voicebox_model = model.voicebox_model
    cond_feature = voicebox_model.cond_emb(cond_codes)
    cond_feature = F.interpolate(
        cond_feature.transpose(1, 2),
        scale_factor=voicebox_model.cond_scale_factor,
    ).transpose(1, 2)
    
    # Run reverse diffusion
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=(device_type == 'cuda')):
        predicted_mel = voicebox_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=model_dict.get('n_timesteps', 15),
            cfg=model_dict.get('cfg', 2.0),
            rescale_cfg=model_dict.get('rescale_cfg', 0.75),
        )
    
    # Vocode mel to wav
    use_decoder_latent = model.use_decoder_latent
    use_decoder_latent_before_agg = model.use_decoder_latent_before_agg
    decoder_latent_pass_transformer = model.decoder_latent_pass_transformer
    
    if use_decoder_latent:
        predicted_audio = model.dualcodec_model.decode_from_latent(predicted_mel.transpose(1, 2), token_lengths)
        return predicted_audio.cpu().squeeze(), 16000
    elif use_decoder_latent_before_agg:
        if decoder_latent_pass_transformer:
            predicted_audio = model.dualcodec_model.dac.decoder(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1, 2))
            predicted_audio = predicted_audio[..., prompt_audio.shape[-1]:]
            return predicted_audio.cpu().squeeze(0), 16000
        else:
            predicted_audio = model.dualcodec_model.dac.decoder(
                model.dualcodec_model.bottleneck_transformer(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1, 2))
            )
            predicted_audio = predicted_audio[..., prompt_audio.shape[-1]:]
            return predicted_audio.cpu().squeeze(0), 16000
    else:
        predicted_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
        return predicted_audio.cpu().squeeze(), 24000


def load_vocoder(device='cuda'):
    """Load Vocos vocoder and mel extractor"""
    print("Loading Vocos model...")
    vocos_model, mel_model = get_vocos_model_spectrogram(device=device)
    vocos_model = vocos_model.to(device)
    infer_vocos = partial(mel_to_wav_vocos, vocos_model)
    return infer_vocos, mel_model


@torch.inference_mode()
def infer_voicebox_librispeech(
    model: VoiceboxWrapper,
    vocoder_decode_func,
    gt_audio_path: Optional[str] = None,
    ref_audio_path: Optional[str] = None,
    gt_audio: Optional[torch.Tensor] = None,
    ref_audio: Optional[torch.Tensor] = None,
    device: str = 'cuda',
    n_timesteps: int = 10,
    cfg: float = 2.0,
    rescale_cfg: float = 0.75,
    merging_threshold: float = 1.0,
):
    """Perform inference using Voicebox model with LibriSpeech data"""
    use_decoder_latent = model.use_decoder_latent
    use_decoder_latent_before_agg = model.use_decoder_latent_before_agg
    decoder_latent_pass_transformer = model.decoder_latent_pass_transformer
    
    # 1. Load ground truth audio and extract semantic codes
    if gt_audio is None:
        if gt_audio_path is None:
            raise ValueError("Either gt_audio_path or gt_audio must be provided")
        print(f"Loading ground truth audio: {gt_audio_path}")
        gt_audio, sr = torchaudio.load(gt_audio_path)
        if sr != 16000:
            gt_audio = torchaudio.transforms.Resample(sr, 16000)(gt_audio)
        if gt_audio.shape[0] > 1:
            gt_audio = gt_audio.mean(dim=0, keepdim=True)
        gt_audio = gt_audio.to(device)
    else:
        gt_audio = gt_audio.to(device)

    # 2. Load reference audio for prompt
    if ref_audio is None:
        if ref_audio_path is None:
            raise ValueError("Either ref_audio_path or ref_audio must be provided")
        print(f"Loading reference audio for prompt: {ref_audio_path}")
        ref_audio, sr = torchaudio.load(ref_audio_path)
        if sr != 16000:
            ref_audio = torchaudio.transforms.Resample(sr, 16000)(ref_audio)
        if ref_audio.shape[0] > 1:
            ref_audio = ref_audio.mean(dim=0, keepdim=True)
        ref_audio = ref_audio.to(device)
    else:
        ref_audio = ref_audio.to(device)

    # 3. Extract semantic codes from reference audio
    print("Extracting semantic codes from reference audio...")
    ref_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(ref_audio.cpu())
    ref_mel_features = ref_mel_features.to(device)
    ref_x_lens = torch.tensor([ref_mel_features.shape[1]], dtype=torch.long, device=device)

    model.dualcodec_model.similarity_threshold = merging_threshold

    ref_dualcodec_output = model._extract_dualcodec_features(ref_audio, mel=ref_mel_features, x_lens=ref_x_lens, manual_threshold=merging_threshold)
    if not use_decoder_latent:
        ref_cond_codes = ref_dualcodec_output['semantic_codes'].squeeze(1)
    else:
        ref_cond_codes = ref_dualcodec_output['semantic_codes_aggregated'].squeeze(1)

    # 4. Extract semantic codes from ground truth audio for conditioning
    print("Extracting semantic codes from ground truth audio...")
    gt_mel_features, _ = feature_extractor_for_dualcodec.extract_fbank(gt_audio.cpu())
    gt_mel_features = gt_mel_features.to(device)
    gt_x_lens = torch.tensor([gt_mel_features.shape[1]], dtype=torch.long, device=device)

    gt_flexicodec_output = model._extract_dualcodec_features(gt_audio, mel=gt_mel_features, x_lens=gt_x_lens)
    gt_cond_codes = gt_flexicodec_output['semantic_codes'].squeeze(1) if not use_decoder_latent else gt_flexicodec_output['semantic_codes_aggregated'].squeeze(1)

    gt_token_lengths = gt_flexicodec_output['token_lengths']

    # 5. Concatenate reference and GT semantic codes (ref first, then GT)
    print("Concatenating reference and GT semantic codes...")
    cond_codes = torch.cat([ref_cond_codes, gt_cond_codes], dim=1)
    
    # 6. Perform inference
    print("Running reverse diffusion...")
    voicebox_model = model.voicebox_model
    
    cond_feature = voicebox_model.cond_emb(cond_codes)
    cond_feature = F.interpolate(
        cond_feature.transpose(1, 2),
        scale_factor=voicebox_model.cond_scale_factor,
    ).transpose(1, 2)
    if model.add_framerate_embedding:
        framerate_idx = model.flex_framerate_options.index(merging_threshold)
        framerate_emb = model.framerate_embedding(torch.tensor(framerate_idx, device=device))
        cond_feature = cond_feature + framerate_emb.unsqueeze(0).unsqueeze(0)
    
    # Use the reference audio as prompt
    if use_decoder_latent:
        prompt_mel = ref_dualcodec_output['decoder_latent'].transpose(1,2)
    elif use_decoder_latent_before_agg:
        prompt_mel = ref_dualcodec_output['decoder_latent_before_agg'].transpose(1,2)
        if decoder_latent_pass_transformer:
            prompt_mel = model.dualcodec_model.bottleneck_transformer(ref_dualcodec_output['decoder_latent_before_agg']).transpose(1,2)
    else:
        prompt_mel = model._extract_mel_features(ref_audio)

    if model.concat_speaker_embedding:
        speaker_embedding = model._extract_speaker_embedding(ref_audio, sample_rate=16000)
        speaker_emb_expanded = speaker_embedding.unsqueeze(0).unsqueeze(0)
        cond_feature = model.spk_linear(torch.cat([cond_feature, speaker_emb_expanded], dim=-1))

    # Use appropriate device type for autocast
    device_type = 'cuda' if device.startswith('cuda') else 'cpu'  # MPS doesn't support autocast, use CPU
    with torch.autocast(device_type=device_type, dtype=torch.float32, enabled=(device_type == 'cuda')):
        predicted_mel = voicebox_model.reverse_diffusion(
            cond=cond_feature,
            prompt=prompt_mel,
            n_timesteps=n_timesteps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
        )

        # 7. Vocode mel to wav
        if use_decoder_latent:
            predicted_audio = model.dualcodec_model.decode_from_latent(predicted_mel.transpose(1,2), gt_token_lengths)
            return predicted_audio.cpu().squeeze(), 16000
        elif use_decoder_latent_before_agg:
            if decoder_latent_pass_transformer:
                predicted_audio = model.dualcodec_model.dac.decoder(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1,2))
                predicted_audio = predicted_audio[...,ref_audio.shape[-1]:]
                return predicted_audio.cpu().squeeze(0), 16000
                # return model.dualcodec_model.dac.decoder(predicted_mel.transpose(1,2)).cpu().squeeze(0), 16000
            else:
                predicted_audio = model.dualcodec_model.dac.decoder(model.dualcodec_model.bottleneck_transformer(torch.cat([prompt_mel, predicted_mel], dim=1).transpose(1,2)))
                predicted_audio = predicted_audio[...,ref_audio.shape[-1]:]
                return predicted_audio.cpu().squeeze(0), 16000
        else:
            print("Vocoding generated mel spectrogram...")
            predicted_audio = vocoder_decode_func(predicted_mel.transpose(1, 2))
            return predicted_audio.cpu().squeeze(), 24000

if __name__ == "__main__":
    # Test/example usage
    checkpoint_path = '/Users/jiaqi/github/FlexiCodec/nartts.safetensors'
    prompt_audio_path = '/Users/jiaqi/github/FlexiCodec/audio_examples/61-70968-0000_ref.wav'
    output_path = '/Users/jiaqi/github/FlexiCodec/61-70968-0000_output.wav'
    
    # Prepare model (loads model and vocoder)
    print("Loading model...")
    model_dict = prepare_voicebox_model(
        checkpoint_path,
        n_timesteps=15,
        cfg=2.0,
        rescale_cfg=0.75
    )
    
    # Load prompt audio
    prompt_audio, _ = torchaudio.load(prompt_audio_path)
    
    # Example: Create dummy audio tokens and length_ids for testing
    # In real usage, these would come from an AR model or codec encoder
    audio_tokens = torch.randint(0, 32768, (100,))  # Example semantic tokens
    length_ids = torch.ones(100, dtype=torch.long)  # Example duration classes
    
    # Run inference
    print("Running inference...")
    output_audio, output_sr = infer_voicebox_tts(
        model_dict=model_dict,
        audio_tokens=audio_tokens,
        length_ids=length_ids,
        prompt_audio=prompt_audio,
        framerate=1.0
    )
    
    # Save output
    if output_sr == 16000:
        torchaudio.save(output_path, output_audio.unsqueeze(0) if output_audio.dim() == 1 else output_audio, output_sr)
    else:
        sf.write(output_path, output_audio.numpy() if isinstance(output_audio, torch.Tensor) else output_audio, output_sr)
    
    print(f"\n✅ Inference complete!")
    print(f"Output saved to: {output_path}")
    print(f"Output sample rate: {output_sr} Hz")
    print(f"Output duration: {output_audio.shape[-1] / output_sr:.2f} seconds")
