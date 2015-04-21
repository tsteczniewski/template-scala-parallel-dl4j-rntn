import predictionio
import argparse
import csv

def import_events(engine_client, file):
  with open(file, 'r') as f:
    reader = csv.DictReader(f, delimiter="\t")
    count = 0
    print "Sending requests..."
    for row in reader:
      print "Expected ", row["Sentiment"], "Phrase", row["Phrase"]
      print "Result:", engine_client.send_query({"phrase": row["Phrase"]})
      count += 1
    print "%s events are sent." % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Send request for all data in training set.")
  parser.add_argument('--url', default="http://localhost:8000")
  parser.add_argument('--file', default="./data/sample_phrase_data.txt")

  args = parser.parse_args()
  print args

  engine_client = \
      predictionio.EngineClient(url="http://localhost:8000", timeout=20)

  import_events(engine_client, args.file)
