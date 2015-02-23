import predictionio
import argparse
import csv

def import_events(client, file):
  with open(file, 'r') as f:
    reader = csv.DictReader(f, delimiter="\t")
    count = 0
    print "Importing data..."
    for row in reader:
      client.create_event(
        event="$set",
        entity_type="phrase",
        entity_id=row["PhraseId"],
        properties= {
          "sentenceId" : row["SentenceId"],
          "phrase" : row["Phrase"],
          "sentiment" : row["Sentiment"],
        }
      )
      count += 1
    print "%s events are imported." % count

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    description="Import sample data for sentiment analysis engine")
  parser.add_argument('--access_key', default='invald_access_key')
  parser.add_argument('--url', default="http://localhost:7070")
  parser.add_argument('--file', default="./data/sample_phrase_data.txt")

  args = parser.parse_args()
  print args

  client = predictionio.EventClient(
    access_key=args.access_key,
    url=args.url,
    threads=5,
    qsize=500)
  import_events(client, args.file)
