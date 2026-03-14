from datetime import datetime

from pydantic import BaseModel, Field


class SensorObservation(BaseModel):
    user_id: str = Field(..., description="User ID")
    timestamp: datetime = Field(default_factory=datetime.now)
    heart_rate: float = Field(..., gt=20, lt=240, description="Current heart rate")
    wrist_acc: list[float] = Field(..., min_length=3, max_length=3, description="Wrist acceleration XYZ")
    wrist_gyro: list[float] = Field(..., min_length=3, max_length=3, description="Wrist gyroscope XYZ")
    ppg_quality: float = Field(1.0, ge=0.0, le=1.0, description="PPG signal quality")
    wrist_bvp: float | None = Field(None, description="Optional wrist BVP sample")
    wrist_eda: float | None = Field(None, ge=0.0, description="Optional wrist EDA sample")
    wrist_temp: float | None = Field(None, ge=10.0, le=45.0, description="Optional wrist temperature sample")
    context_override: str | None = Field(
        None,
        description="Optional context from upstream: resting/walking/running/cycling/stairs/household",
    )
