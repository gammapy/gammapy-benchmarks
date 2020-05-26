"""Check all catalogs and sources."""
import logging

from gammapy.catalog import SOURCE_CATALOGS
from gammapy.modeling.models import (SkyModel, SkyModels, SpatialModel,
                                     SpectralModel)

log = logging.getLogger(__name__)


def check_source(source):
    str(source)
    # source.energy_range
    # source.spectral_model_type
    assert isinstance(source.spectral_model(), (SpectralModel, type(None)))
    # source.spatial_model_type
    assert isinstance(source.sky_model(), (SkyModel, SkyModels))
    # source.flux_points


def main():
    for name in SOURCE_CATALOGS:
        log.info(f"Checking catalog: {name}")
        catalog = SOURCE_CATALOGS[name]()
        for source in catalog:
            if source.index not in set(range(0, 10)):
                continue
            log.info(f"Checking source: {source.index}, {source.name}")
            try:
                check_source(source)
            except Exception as exc:
                log.exception(exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
