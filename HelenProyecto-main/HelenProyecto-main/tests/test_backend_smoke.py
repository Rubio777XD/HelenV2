from backendHelen import server


def test_backend_smoke_without_camera():
    config = server.RuntimeConfig(enable_camera=False, fallback_to_synthetic=True)
    runtime = server.HelenRuntime(config=config)
    try:
        runtime.start()
    finally:
        runtime.stop(export_report=False)
