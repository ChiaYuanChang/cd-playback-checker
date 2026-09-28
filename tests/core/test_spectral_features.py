import numpy as np

from compare_audio.core.spectral_features import (
    FeatureExtractor,
    FeatureSet,
    extract_features,
    mel_filterbank,
)


def test_streaming_matches_batch(config):
    rng = np.random.default_rng(3)
    signal = rng.normal(0, 0.1, 50_000).astype(np.float32)
    batch = extract_features(signal, config)

    extractor = FeatureExtractor(config)
    pieces, position = [], 0
    for size in [1, 999, 160, 4321, 17, 20_000, 30_000]:
        pieces.append(extractor.push(signal[position : position + size]))
        position += size
    streamed = FeatureSet.concatenate(pieces, config.n_mels)

    np.testing.assert_allclose(streamed.log_mel, batch.log_mel, atol=1e-3)
    np.testing.assert_allclose(streamed.band_env_db, batch.band_env_db, atol=1e-3)
    np.testing.assert_array_equal(streamed.wave_max, batch.wave_max)


def test_frame_counts(config):
    signal = np.zeros(16000, np.float32)
    features = extract_features(signal, config)
    assert len(features.log_mel) == 1 + (16000 - config.n_fft) // config.hop
    assert len(features.band_env_db) == 16000 // config.envelope_hop


def test_mel_filters_cover_every_band(config):
    bank = mel_filterbank(config)
    assert bank.shape == (config.n_fft // 2 + 1, config.n_mels)
    assert np.all(bank.sum(axis=0) > 0)
