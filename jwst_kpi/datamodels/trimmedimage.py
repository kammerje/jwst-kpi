from jwst.datamodels import ImageModel


class TrimmedImageModel(ImageModel):
    """
    A data model for KPFITS files with extracted kernel phases.

    Inherits all properties from default JWST images,
    with additional metadata for KP pipeline.
    """

    schema_url = "http://stsci.edu/schemas/jwst_kpi_datamodel/trimmedimage.schema"
