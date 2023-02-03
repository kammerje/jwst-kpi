from jwst.datamodels import CubeModel


class BadPixCubeModel(CubeModel):
    schema_url = "http://stsci.edu/schemas/jwst_kpi_datamodel/badpixcube.schema"

    def __init__(self, init=None, **kwargs):

        super(BadPixCubeModel, self).__init__(init=init, **kwargs)

        # Implicitly create remaining array
        self.dq_mod = self.dq_mod
