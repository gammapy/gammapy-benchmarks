from gammapy.modeling import Parameter, Parameters
from gammapy.modeling.models import SpectralModel


class EnergyBiasSpectralModel(SpectralModel):
    r"""Spectral model with energy bias.

    Parameters
    ----------
    spectral_model : `SpectralModel`
        Spectral model.
    name : str, optional
        parameter name
    bias: float
        Norm of the energy bias
    """
    tag = "EnergyBiasSpectralModel"

    def __init__(
        self, spectral_model, name="bias", bias=0.0,
    ):
        self.spectral_model = spectral_model
        self.parameter_name = name
        self.bias_parameter = Parameter(
            name, bias, unit="", min=-1.0, max=2.0, frozen=False
        )
        parameters = Parameters([self.bias_parameter])

        super()._init_from_parameters(parameters)

    @property
    def parameters(self):
        return self._parameters + self.spectral_model.parameters

    def evaluate(self, energy, **kwargs):
        """Evaluate the model at a given energy."""
        # assign redshift value and remove it from dictionary
        # since it does not belong to the spectral model
        del kwargs[self.parameter_name]
        new_energy = (1 + self.bias_parameter.value) * energy
        dnde = self.spectral_model.evaluate(energy=new_energy, **kwargs)
        return dnde

    def to_dict(self):
        return {
            "type": self.tag,
            "base_model": self.spectral_model.to_dict(),
            "bias_parameter": {
                "name": self.parameter_name,
                "value": self.bias_parameter.value,
            },
            "parameters": self._parameters.to_dict()["parameters"],
        }

    @classmethod
    def from_dict(cls, data):
        from gammapy.modeling.models import SPECTRAL_MODELS

        model_class = SPECTRAL_MODELS.get_cls(data["base_model"]["type"])
        model = cls(
            spectral_model=model_class.from_dict(data["base_model"]),
            bias=data["bias_parameter"]["value"],
            name=data["bias_parameter"]["name"],
        )
        model._update_from_dict(data)
        return model
