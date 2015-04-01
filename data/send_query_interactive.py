import sys
import predictionio

engine_client = \
  predictionio.EngineClient(url="http://localhost:8000", timeout=20)

while True:
  line = sys.stdin.readline()
  print "Result:", engine_client.send_query({"phrase": line.rstrip()})

engine_client.close()
