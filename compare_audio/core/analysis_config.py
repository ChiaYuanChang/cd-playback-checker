"""Tunable parameters of the alignment and detection pipeline."""

from pydantic import BaseModel, ConfigDict, Field


class AnalysisConfig(BaseModel):
    """All thresholds live here so a test profile can override them in one place."""

    model_config = ConfigDict(frozen=True)

    # --- signal representation -------------------------------------------------
    sample_rate: int = Field(default=16000, description="Analysis sample rate (Hz).")
    n_fft: int = Field(default=1024, description="STFT size in samples.")
    hop: int = Field(default=160, description="STFT hop in samples (10 ms at 16 kHz).")
    n_mels: int = Field(default=48, description="Number of mel bands.")
    fmin_hz: float = Field(default=150.0, description="Lowest mel band edge (Hz).")
    fmax_hz: float = Field(default=6000.0, description="Highest mel band edge (Hz).")
    ref_floor_db: float = Field(
        default=70.0,
        description="Reference features are clamped to (track max - this) dB.",
    )
    envelope_rate_hz: int = Field(
        default=200, description="Rate of the fine energy/waveform envelopes (Hz)."
    )
    envelope_band_hz: tuple[float, float] = Field(
        default=(300.0, 5000.0),
        description="Band used for the fine energy envelope (dropout detection).",
    )

    sensitivity: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Recognition sensitivity; 50 preserves the profile thresholds.",
    )

    # --- tracking ------------------------------------------------------------
    window_s: float = Field(default=1.5, description="Length of one matching window.")
    step_s: float = Field(default=0.25, description="Tracker step between windows.")
    search_radius_s: float = Field(
        default=2.0, description="Local search radius around the predicted position."
    )
    match_threshold: float = Field(
        default=0.45, description="NCC score needed to accept a window as matched."
    )
    acquire_threshold: float = Field(
        default=0.55, description="NCC score needed to (re)acquire lock globally."
    )
    lost_steps: int = Field(
        default=3, description="Consecutive unmatched steps before a global search."
    )
    jump_min_loudness_db: float = Field(
        default=12.0,
        description="A window must be this far above the noise floor to move the lock.",
    )
    coarse_factor: int = Field(
        default=4, description="Time decimation of features for the global search."
    )
    global_candidates: int = Field(
        default=5, description="Coarse peaks refined during a global search."
    )

    # --- event detection -------------------------------------------------------
    offset_tolerance_s: float = Field(
        default=0.04, description="Offset change smaller than this is not a jump."
    )
    min_gap_s: float = Field(
        default=0.08,
        description="Unexplained audio shorter than this counts as an instant jump.",
    )
    silence_margin_db: float = Field(
        default=6.0,
        description="Frames within this margin of the noise floor are silent.",
    )
    dropout_min_s: float = Field(
        default=0.03, description="Shortest mute reported as a dropout."
    )
    dropout_depth_db: float = Field(
        default=10.0, description="Level deficit (vs expected) that marks a dropout."
    )
    uncertain_min_s: float = Field(
        default=1.5, description="Shortest unmatched stretch with sound reported."
    )
    track_gap_max_s: float = Field(
        default=6.0, description="Longest silence between tracks treated as normal."
    )
    edge_tolerance_s: float = Field(
        default=1.5,
        description="Audible track start/end may be missed by this much without error.",
    )

    @property
    def effective_match_threshold(self) -> float:
        return min(0.95, max(0.2, self.match_threshold - (self.sensitivity - 50) / 500))

    @property
    def effective_acquire_threshold(self) -> float:
        return min(
            0.99,
            max(
                self.effective_match_threshold,
                self.acquire_threshold - (self.sensitivity - 50) / 500,
            ),
        )

    @property
    def effective_silence_margin_db(self) -> float:
        return max(0.0, self.silence_margin_db - (self.sensitivity - 50) * 0.06)

    @property
    def effective_jump_min_loudness_db(self) -> float:
        return max(0.0, self.jump_min_loudness_db - (self.sensitivity - 50) * 0.12)

    def recognition_thresholds(self) -> dict[str, float]:
        return {
            "match_threshold": self.effective_match_threshold,
            "acquire_threshold": self.effective_acquire_threshold,
            "silence_margin_db": self.effective_silence_margin_db,
            "jump_min_loudness_db": self.effective_jump_min_loudness_db,
        }

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop

    def seconds_to_frames(self, seconds: float) -> int:
        return int(round(seconds * self.frame_rate))

    @property
    def window_frames(self) -> int:
        return self.seconds_to_frames(self.window_s)

    @property
    def step_frames(self) -> int:
        return max(1, self.seconds_to_frames(self.step_s))

    @property
    def envelope_hop(self) -> int:
        return self.sample_rate // self.envelope_rate_hz

    def feature_signature(self) -> str:
        """Identifies the parameters that change cached reference features."""
        return (
            f"sr{self.sample_rate}-fft{self.n_fft}-hop{self.hop}-mel{self.n_mels}"
            f"-{self.fmin_hz:g}-{self.fmax_hz:g}-floor{self.ref_floor_db:g}"
            f"-env{self.envelope_rate_hz}-{self.envelope_band_hz[0]:g}"
            f"-{self.envelope_band_hz[1]:g}"
        )
