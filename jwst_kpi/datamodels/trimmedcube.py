from jwst.datamodels import CubeModel


class TrimmedCubeModel(CubeModel):
    """
    A data model for KPFITS files with extracted kernel phases.

    Inherits all properties from default JWST cubes,
    with additional metadata for KP pipeline.
    """

    schema_url = "http://stsci.edu/schemas/jwst_kpi_datamodel/trimmedcube.schema"
