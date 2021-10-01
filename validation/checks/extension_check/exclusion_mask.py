from astropy.coordinates import Angle 
from regions import PointSkyRegion, CircleSkyRegion
from gammapy.catalog import CATALOG_REGISTRY, SourceCatalogGammaCat

def build_exclusion_mask_from_catalog(geom, catalog="gamma-cat", min_radius="0.3 deg"):
    """Build exclusion mask from a given geometry and input catalog.
    
    Parameters:
    -----------
    geom : `~gammapy.maps.WcsGeom`
        the input WCS geometry
    catalog : str or `~gammapy.catalog.SourceCatalog`
        the input catalog to use. Default is 'gamma-cat'.
    min_radius : `~astropy.coordinates.Angle`
        the minimum radius of the circular exclusion radius around a source. Default is 0.3 deg.
    """
    
    if catalog is None:
        catalog = "gamma-cat"
    
    if isinstance(catalog, str):       
        try:
            catalog = CATALOG_REGISTRY.get_cls(catalog)()
        except KeyError:
            raise ValueError(f"Unknown catalog name. Available catalogs are :\n{CATALOG_REGISTRY}")

    exclusion_regions = []
    min_radius = Angle(min_radius)
    
    is_in_geom = geom.contains(catalog.positions)
    src_in_geom = catalog[is_in_geom].to_models()
    
    for src in src_in_geom.to_regions():
        if not isinstance(src, PointSkyRegion):
            radius += src.width
        
        exclusion_regions.append(CircleSkyRegion(src.center, min_radius))
    
    return ~geom.region_mask(exclusion_regions)

