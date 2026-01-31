from Intents.Intent_predictor import predict_intent, handled_command
from Entities.Entity_extractor import extract_entities

command = "Wake up"
intent, conf = predict_intent(command)
entities = extract_entities(intent, command)
while True:
    handled_command(command)
    break
