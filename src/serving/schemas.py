from pydantic import BaseModel, Field, ConfigDict

# ---------------------------------------------------------
# Request schema
# ---------------------------------------------------------
class WineFeatures(BaseModel):
    """Input features for wine quality prediction."""

    model_config = ConfigDict(populate_by_name=True)

    fixed_acidity: float = Field(alias="fixed acidity")
    volatile_acidity: float = Field(alias="volatile acidity")
    citric_acid: float = Field(alias="citric acid")
    residual_sugar: float = Field(alias="residual sugar")
    chlorides: float = Field(alias="chlorides")
    free_sulfur_dioxide: float = Field(alias="free sulfur dioxide")
    total_sulfur_dioxide: float = Field(alias="total sulfur dioxide")
    density: float = Field(alias="density")
    pH: float = Field(alias="pH")
    sulphates: float = Field(alias="sulphates")
    alcohol: float = Field(alias="alcohol")


# ---------------------------------------------------------
# Response schema
# ---------------------------------------------------------

class PredictionResponse(BaseModel):
    """Prediction response returned by the API.

    JSON response schema:
    - predicted_quality: predicted wine quality
    - model_name: registered model name in MLflow
    - model_version: loaded model version
    """

    predicted_quality: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    """Health check response returned by the API.

    JSON response schema:
    - status: service status
    - model_name: registered model name in MLflow
    - model_version: loaded model version
    """

    status: str
    model_name: str
    model_version: str