import predictionio

engine_client = \
    predictionio.EngineClient(url="http://localhost:8000", timeout=20)
print engine_client.send_query({"phrase": "goose"})
engine_client.close()
