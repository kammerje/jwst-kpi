from unittest.mock import patch


def test_svo():
    from jwst_kpi import constants

    assert constants.wave_nircam is not None


def test_no_network(caplog):
    import logging
    with caplog.at_level(logging.WARNING):
        with patch("jwst_kpi.utils.has_network_access", return_value=False):
            # Re-import or reload the module to trigger the fallback
            import importlib
            from jwst_kpi import constants

            importlib.reload(constants)

            # Your test assertions here
            assert constants.wave_nircam is not None
            assert "No network access" in caplog.text
